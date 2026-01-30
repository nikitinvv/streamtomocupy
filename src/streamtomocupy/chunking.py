import cupy as cp
import numpy as np
import os
from .utils import *
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

class Chunking:
    def __init__(self, nbytes, ngpus):
        # prepare containers
        self.stream = [[None for _ in range(3)] for _ in range(ngpus)]
        self.pinned_mem = [None for _ in range(ngpus)]
        self.gpu_mem = [None for _ in range(ngpus)]
        self.pool_inp = [None for _ in range(ngpus)]
        self.pool_out = [None for _ in range(ngpus)]

        # thread sizing for io operations for each gpu
        per_gpu_threads = os.cpu_count() // 2 // ngpus

        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                # pinned and device memory allocations
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)

                # create dedicated streams (non_blocking for overlaps)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=True)

                # per-GPU thread pools for IO/CPU-side work
                self.pool_inp[igpu] = ThreadPoolExecutor(max_workers=per_gpu_threads)
                self.pool_out[igpu] = ThreadPoolExecutor(max_workers=per_gpu_threads)

        # pool to run different gpus
        self.pool = ThreadPoolExecutor(ngpus)
        self.ngpus = ngpus
            

    def gpu_batch(self, chunk, axis_out=0, axis_inp=0, nout=1):
        """
        Multigpu processing of functions with syntax f(out1_proper,out2_proper,..out1_nonproper,out2_nonproper,...
        inp1_proper,inp2_proper,..inp1_nonproper,inp2_nonproper,...,inp1,inp2..)

        where
        out*_proper are output numpy arrays (nout arrays) of proper for chunking shape: shape[axis_out] = X, where X is the chunking axis size,
        inp*_proper are input numpy arrays of proper for chunking shape: shape[axis_inp] = X, where X is the chunking axis size,
        out*_nonproper are output cupy arrays of nonproper shape (not X) for chunkning size. These arrays are on GPUs and filled directly without CPU-GPU transfers
        inp*_nonproper are input cupy/numpy arrays of nonproper shape for (not X) chunkning size. These arrays are stored on one GPU and distributed to others inside this function
        inp* are non-array input parameters
        """

        def decorator(func):
            def inner(*args):
                # if no chunking the just run on the current gpu
                if not any(isinstance(a, np.ndarray) for a in args):
                    func(*args)
                    return

                # extract inputs and outputs
                cl = args[0]
                out = args[1 : 1 + nout]
                inp = args[1 + nout :]

                # size of the chnunking dimension
                size = inp[0].shape[axis_inp]
                # adjust the number of gpus, some may not be needed
                ngpus_adj = min(int(np.ceil(size / chunk)), self.ngpus)
                # sizes per gpu
                gsize = int(np.ceil(size / ngpus_adj))

                # proper_inp is the number of variables of type numpy that have proper_inp shape size in the chunking dimension
                # non proper_inp is the number of variables of type numpy/cupy that have nonproper_inp shape size
                proper_inp, nonproper_inp = 0, 0
                proper_out, nonproper_out = 0, 0
                
                for k in range(0, len(out)):
                    if (isinstance(out[k], np.ndarray)) and len(out[k].shape) > axis_out + 1 and out[k].shape[axis_out] == size:
                        proper_out += 1
                    elif isinstance(out[k], list):
                        nonproper_out += 1

                for k in range(0, len(inp)):
                    if (isinstance(inp[k], np.ndarray)) and len(inp[k].shape) > axis_inp + 1 and inp[k].shape[axis_inp] == size:
                        # cpu arrays of the proper_inp shape for processing by chunks
                        proper_inp += 1
                    elif isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray):
                        # arrays of nonproper_inp shape for processing by chunks
                        nonproper_inp += 1
                # thread pool for each gpu
                futures = []
                for igpu in range(ngpus_adj):
                    ## parse inp and out variables, place chunk in lists
                    ginp, gout = [], []
                    sl = slice(igpu * gsize, (igpu + 1) * gsize)
                    for x in inp[:proper_inp]:
                        idx = [slice(None)] * 3
                        idx[axis_inp] = sl
                        ginp.append(x[tuple(idx)])
                    ginp.extend(inp[proper_inp:])

                    for x in out[:proper_out]:
                        idx = [slice(None)] * 3
                        idx[axis_out] = sl
                        gout.append(x[tuple(idx)])
                    gout.extend([x[igpu] for x in out[proper_out:]])

                    if np.prod(gout[0].shape) == 0:  ## nothing to work on for this gpu
                        break

                    # regular parallelization case with the same dimension size for inp and out
                    futures.append(
                        self.pool.submit(
                            self.run, cl, gout, ginp, proper_inp, nonproper_inp, proper_out, nonproper_out, chunk, axis_out, axis_inp, func, igpu
                        )
                    )

                done, notdone = wait(futures)
                for f in done:
                    exc = f.exception()
                    if exc is not None:
                        raise exc

                # switch to 0 gpu
                cp.cuda.Device(0).use()

            return inner

        return decorator

    def run(self, cl, out, inp, proper_inp, nonproper_inp, proper_out, nonproper_out, chunk, axis_out, axis_inp, func, igpu):
        """Run by chunks, the case where the size of chunking dimension is the same for inp and out"""

        # set gpu and get references to pinned and gpu memory, and streams
        cp.cuda.Device(igpu).use()
        pinned_mem = self.pinned_mem[igpu]
        gpu_mem = self.gpu_mem[igpu]
        stream = self.stream[igpu]
        pool_inp = self.pool_inp[igpu]
        pool_out = self.pool_out[igpu]

        size = inp[0].shape[axis_inp]
        nchunk = int(np.ceil(size / chunk))

        # allocate pinned and gpu memory buffers
        out_pinned, out_gpu, offset = self.alloc_double_buffers(out[:proper_out], axis_out, pinned_mem, gpu_mem, 0, chunk)
        inp_pinned, inp_gpu, offset = self.alloc_double_buffers(inp[:proper_inp], axis_inp, pinned_mem, gpu_mem, offset, chunk)

        # if any proper_inp:nonproper_inp array is numpy, copy it to gpu
        for k in range(proper_inp, proper_inp + nonproper_inp):
            inp[k] = cp.asarray(inp[k])
        
       
        
        def c2p(buf_id, k):
            st = k * chunk
            end = min(size, (k + 1) * chunk)
            
            src = self.mk_slices(axis_inp, slice(st, end))
            dst = self.mk_slices(axis_inp, slice(0, end-st))

            return [
                self.copy(inp[j][src], inp_pinned[buf_id][j][dst], pool_inp)
                for j in range(proper_inp)
            ]


        def p2c(buf_id, k):
            st = k * chunk
            end = min(size, (k + 1) * chunk)
            
            src = self.mk_slices(axis_out, slice(0, end-st))
            dst = self.mk_slices(axis_out, slice(st, end))

            return [
                self.copy(out_pinned[buf_id][j][src], out[j][dst], pool_out)
                for j in range(proper_out)
            ]


        def p2g(buf_id):
            for j in range(proper_inp):
                inp_gpu[buf_id][j].set(inp_pinned[buf_id][j])


        def g2p(buf_id):
            for j in range(proper_out):
                out_gpu[buf_id][j].get(out=out_pinned[buf_id][j], blocking=False)


        def p(buf_id, k):
            # st = k * chunk
            # end = min(size, (k + 1) * chunk)
            
            # inp_gpu_c = self.slice_bufs(inp_gpu[buf_id], axis_inp, end - st)
            # out_gpu_c = self.slice_bufs(out_gpu[buf_id], axis_out, end - st)
            
            # use whole buffer instead:
            inp_gpu_c = inp_gpu[buf_id]
            out_gpu_c = out_gpu[buf_id]

            func(
                cl,
                *out_gpu_c,
                *out[proper_out:],
                *inp_gpu_c,
                *inp[proper_inp : proper_inp + nonproper_inp],
                *inp[proper_inp + nonproper_inp :],
            )


        # Pipeline processing
        th_inp, th_out = [], []
        
        # run by chunks, overlap data transfers and computations
        for k in range(nchunk + 4):
            if k < nchunk:
                th_inp = c2p(k % 2, k) 
            
            if 0 < k < nchunk + 1:
                with stream[(k - 1) % 3]:
                    p2g((k - 1) % 2)

            if 1 < k < nchunk + 2:
                with stream[(k - 2) % 3]:
                    p((k - 2) % 2, k - 2)

            if 2 < k < nchunk + 3:
                with stream[(k - 3) % 3]:
                    g2p((k - 3) % 2)

            if k > 3: 
                th_out = p2c((k - 4) % 2, k - 4) 

            # Sync
            for s in stream:
                s.synchronize()
            for f in th_inp:
                wait(f)
            for f in th_out:
                wait(f)

            

    def alloc_double_buffers(self, arrs, axis, pinned_mem, gpu_mem, offset, chunk):
        """Allocate pinned and gpu memory buffers for each chunk for each variable (double for streaming)"""

        pinned = [[], []]
        gpu = [[], []]

        for j in (0, 1):  # double buffer
            for a in arrs:
                shape0 = list(a.shape)
                shape0[axis] = chunk
                shape0 = tuple(shape0)

                n = int(np.prod(shape0))
                itemsize = np.dtype(a.dtype).itemsize
                nbytes = n * itemsize

                try:
                    pinned[j].append(np.frombuffer(pinned_mem + offset, a.dtype, n).reshape(shape0))
                    gpu[j].append(cp.ndarray(shape0, dtype=a.dtype, memptr=gpu_mem + offset))
                except Exception as e:
                    raise RuntimeError(f"Failed to allocate pinned/gpu buffers") from e

                offset += nbytes

        return pinned, gpu, offset
    
    def copy(self, u, out, pool):
        """Parallel array copy, numpy is too slow"""

        def _copy(out, u, st, end):
            out[st:end] = u[st:end]
            return out

        nthreads = min(pool._max_workers, u.shape[0])
        nchunk = int(np.ceil(u.shape[0] / nthreads))
        futures = [pool.submit(_copy, out, u, k * nchunk, min((k + 1) * nchunk, u.shape[0])) for k in range(nthreads)]
        return futures

    ####################### Slicing       #########################
    def slice_bufs(self, bufs, axis, n):
        """Slicing functon 1"""

        slc = [slice(None)] * 3
        slc[axis] = slice(0, n)
        return [b[tuple(slc)] for b in bufs]

    def mk_slices(self, axis, sl):
        """Slicing functon 2"""

        res = [slice(None)] * 3
        res[axis] = sl
        return tuple(res)
