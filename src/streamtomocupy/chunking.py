import cupy as cp
import numpy as np
from streamtomocupy.utils import copy
from concurrent.futures import ThreadPoolExecutor, wait


def gpu_batch(chunk=8, ngpus=1, axis_out=0, axis_inp=0):
    def decorator(func):
        def inner(*args):
            cl = args[0]
            out = args[1]
            inp = args[2:]

            size = out.shape[axis_out]
            ngpus_adj = min(int(np.ceil(size / chunk)), ngpus)
            gsize = int(np.ceil(size / ngpus_adj))

            proper = 0
            nonproper = 0
            for k in range(0, len(inp)):
                if (
                    (isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray))
                    and len(inp[k].shape) > axis_inp + 1
                    and inp[k].shape[axis_inp] == size
                ):
                    # arrays of the proper shape for processing by chunks
                    proper += 1
                elif isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray):
                    # arrays of nonproper shape for processing by chunks
                    nonproper += 1

            pool = ThreadPoolExecutor(ngpus_adj)
            futures = []
            for igpu in range(ngpus_adj):
                if axis_out == 0:
                    gout = out[igpu * gsize : (igpu + 1) * gsize]
                if axis_out == 1:
                    gout = out[:, igpu * gsize : (igpu + 1) * gsize]
                if np.prod(gout.shape) == 0:
                    break
                if axis_inp == 0:
                    ginp = [x[igpu * gsize : (igpu + 1) * gsize] for x in inp[:proper]]
                    if len(inp[proper:]) > 0:
                        ginp.extend(inp[proper:])
                if axis_inp == 1:
                    ginp = [
                        x[:, igpu * gsize : (igpu + 1) * gsize] for x in inp[:proper]
                    ]
                    if len(inp[proper:]) > 0:
                        ginp.extend(inp[proper:])

                futures.append(
                    pool.submit(
                        run,
                        gout,
                        ginp,
                        chunk,
                        proper,
                        nonproper,
                        axis_out,
                        axis_inp,
                        cl,
                        func,
                        igpu,
                        ngpus_adj,
                    )
                )
            wait(futures)

        return inner

    return decorator


def run(out, inp, chunk, proper, nonproper, axis_out, axis_inp, cl, func, igpu, ngpus):
    cp.cuda.Device(igpu).use()

    pinned_mem = cl.pinned_mem[igpu]
    gpu_mem = cl.gpu_mem[igpu]
    stream = cl.stream[igpu]

    size = out.shape[axis_out]
    nchunk = int(np.ceil(size / chunk))
    out_shape0 = list(out.shape)
    out_shape0[axis_out] = chunk

    # take memory from the buffer
    out_gpu = cp.ndarray([2, *out_shape0], dtype=out.dtype, memptr=gpu_mem)
    out_pinned = np.frombuffer(
        pinned_mem, out.dtype, np.prod([2, *out_shape0])
    ).reshape([2, *out_shape0])

    # shift memory pointer
    offset = np.prod([2, *out_shape0]) * np.dtype(out.dtype).itemsize
    # determine the number of inputs and allocate memory for each input chunk
    inp_gpu = [[], []]
    inp_pinned = [[], []]

    for j in range(2):  # do it twice to assign memory pointers
        for k in range(proper):
            inp_shape0 = list(inp[k].shape)
            inp_shape0[axis_inp] = chunk

            # take memory from the buffers
            inp_gpu[j].append(
                cp.ndarray(inp_shape0, dtype=inp[k].dtype, memptr=gpu_mem + offset)
            )
            inp_pinned[j].append(
                np.frombuffer(
                    pinned_mem + offset, inp[k].dtype, np.prod(inp_shape0)
                ).reshape(inp_shape0)
            )

            # shift memory pointer
            offset += np.prod(inp_shape0) * np.dtype(inp[k].dtype).itemsize

    for k in range(proper, proper + nonproper):
        inp[k] = cp.asarray(inp[k])

    pool_inp = ThreadPoolExecutor(16 // ngpus)
    pool_out = ThreadPoolExecutor(16 // ngpus)
    # proper=1
    # run by chunks
    for k in range(nchunk + 2):
        if k > 0 and k < nchunk + 1:
            with stream[1]:  # processing
                func(cl, out_gpu[(k - 1) % 2], *inp_gpu[(k - 1) % 2], *inp[proper:])

        if k > 1:
            with stream[2]:  # gpu->cpu copy
                out_gpu[(k - 2) % 2].get(
                    out=out_pinned[(k - 2) % 2], blocking=False
                )  ####Note blocking parameter is not define for old cupy versions

        if k < nchunk:
            with stream[0]:  # copy to pinned memory
                st, end = k * chunk, min(size, (k + 1) * chunk)
                for j in range(proper):
                    if axis_inp == 0:
                        copy(inp[j][st:end], inp_pinned[k % 2][j][: end - st], pool_inp)
                    elif axis_inp == 1:
                        copy(
                            inp[j][:, st:end],
                            inp_pinned[k % 2][j][:, : end - st],
                            pool_inp,
                        )

                with stream[0]:  # cpu->gpu copy
                    for j in range(proper):
                        inp_gpu[k % 2][j].set(inp_pinned[k % 2][j])
        stream[2].synchronize()
        if k > 1:
            st, end = (k - 2) * chunk, min(size, (k - 1) * chunk)
            if axis_out == 0:
                copy(out_pinned[(k - 2) % 2][: end - st], out[st:end], pool_out)
            if axis_out == 1:
                copy(out_pinned[(k - 2) % 2][:, : end - st], out[:, st:end], pool_out)
        stream[0].synchronize()
        stream[1].synchronize()
    return
