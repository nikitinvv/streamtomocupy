import cupy as cp
import numpy as np
from concurrent.futures import wait


def _copy(res, u, st, end):
    res[st:end] = u[st:end]


def copy(u, res, pool):
    nthreads = pool._max_workers
    nthreads = min(nthreads, u.shape[0])
    nchunk = int(np.ceil(u.shape[0] / nthreads))
    futures = [
        pool.submit(_copy, res, u, k * nchunk, min((k + 1) * nchunk, u.shape[0]))
        for k in range(nthreads)
    ]
    wait(futures)
    return res


place_kernel = cp.RawKernel(
    r"""                            
    extern "C"                
    void __global__ place(float* f, int n0, int n1, int n2)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n0 || ty >= n1 || tz >= n2)
            return;
        int ind = tz*n0*n1+ty*n0+tx;
        if (f[ind]<=0)
            f[ind] = 1;    
        f[ind] = -log(f[ind]);
        if (isnan(f[ind]))
            f[ind] = 6;                    
        if (isinf(f[ind]))
            f[ind] = 0;                                   
    }
    """,
    "place",
)


conv2d_kernel = cp.RawKernel(
    r"""                            
    extern "C"                
    void __global__ conv2d(float* out, float* x, float* w, int stride0, int stride1, 
                             int b, int co, int ho, int wo,
                             int ci, int hi, int wi, 
                             int groups, int chunk, int chunko,
                             int hk, int wk)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= wo || ty >= ho || tz >= b)
            return;

        int iii,jjj;
        int indx, indw,indo;
        float outt;
        for (int g=0; g<groups;g++)
        for (int igo=g*chunko; igo<(g+1)*chunko;igo++)
        {
            indo = tz*co*ho*wo + igo*ho*wo + ty*wo+tx;                             
            outt = 0;
            for (int ig=g*chunk; ig<(g+1)*chunk;ig++)            
            for (int ii=0; ii<hk; ii++)
            for (int jj=0; jj<wk; jj++)
            {                
                indw = igo*hk*wk+ii*wk+jj;                                    
                iii = ii+ty*stride0;        
                jjj = jj+tx*stride1;                    
                indx = tz*ci*hi*wi + ig*hi*wi + iii*wi+jjj;
                outt += x[indx]*w[indw];                    
            }
            out[indo] = outt;
        }
    }
    """,
    "conv2d",
)

# conv_transpose2d_kernel = cp.RawKernel(r'''
#     extern "C"
#     void __global__ conv2dtranspose(float* out, float* x, float* w, int stride0, int stride1,
#                              int b, int co, int ho, int wo,
#                              int ci, int hi, int wi,
#                              int groups, int chunk, int chunko,
#                              int hk, int wk)
#     {
#         int tx = blockDim.x * blockIdx.x + threadIdx.x;
#         int ty = blockDim.y * blockIdx.y + threadIdx.y;
#         int tz = blockDim.z * blockIdx.z + threadIdx.z;
#         if (tx >= wo || ty >= ho || tz >= b)
#             return;


#         int iii,jjj;
#         int indx, indw,indo;
#         float xt;
#         float v;
#         for (int g=0; g<groups;g++)
#         for (int igo=g*chunko; igo<(g+1)*chunko;igo++)
#         {
#             indx = tz*co*ho*wo + igo*ho*wo + ty*wo+tx;
#             xt = x[indx];

#             for (int ig=g*chunk; ig<(g+1)*chunk;ig++)
#             for (int ii=0; ii<hk; ii++)
#             for (int jj=0; jj<wk; jj++)
#             {
#                 iii = ii+ty*stride0;
#                 jjj = jj+tx*stride1;
#                 indo = tz*ci*hi*wi + ig*hi*wi + iii*wi+jjj;
#                 indw = igo*ci*hk*wk+ig*hk*wk+ii*wk+jj;
#                 v = xt*w[indw];
#                 //out[indo] += v;
#                 atomicAdd(&(out[indo]), v);
#             }
#         }
#     }
#     ''', 'conv2dtranspose')

conv_transpose2d_kernel = cp.RawKernel(
    r"""                            
    extern "C"                
    void __global__ conv2dtranspose(float* out, float* x, float* w, int stride0, int stride1, 
                             int b, int co, int ho, int wo,
                             int ci, int hi, int wi, 
                             int groups, int chunk, int chunko,
                             int hk, int wk, int bs)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= wo || ty >= ho || tz >= b)
            return;        
        
        int iii,jjj;
        int indx, indw,indo, indt;
        float xt;
        float v;
        
        int stx,sty,endx,endy,chunkx,chunky;              

        chunky = ceil((hk+bs*stride0)/(float)bs);
        sty = threadIdx.y*chunky;
        endy = min(sty+chunky, hk+bs*stride0);                                        
        chunkx = ceil((wk+bs*stride1)/(float)bs);
        stx = threadIdx.x*chunkx;
        endx = min(stx+chunkx, wk+bs*stride1);      
                                       
        indt = threadIdx.z*(blockDim.y+hk*stride0)*(blockDim.x+wk*stride1);
        extern __shared__ float s[];
        
        for (int g=0; g<groups;g++)
        for (int igo=g*chunko; igo<(g+1)*chunko;igo++)
        {       
            indx = tz*co*ho*wo + igo*ho*wo + ty*wo+tx;                                  
            xt = x[indx];                                                   
            
            for (int ig=g*chunk; ig<(g+1)*chunk;ig++)                        
            {
                for (int ii=sty; ii<endy; ii++)
                    for (int jj=stx; jj<endx; jj++)
                        s[indt+ii*(blockDim.x+wk*stride1)+jj] = 0;
                __syncthreads();

                for (int ii=0; ii<hk; ii++)
                for (int jj=0; jj<wk; jj++)
                {                                
                    indw = igo*ci*hk*wk+ig*hk*wk+ii*wk+jj;                                                    
                    iii = ii+threadIdx.y*stride0;
                    jjj = jj+threadIdx.x*stride1;
                    v = xt*w[indw];                    
                    atomicAdd(&(s[indt+iii*(blockDim.x+wk*stride1)+jjj]), v);                    
                }   
                __syncthreads();
                                                                                         
                for (int ii=sty; ii<endy; ii++)
                    for (int jj=stx; jj<endx; jj++)
                    {                                   
                        iii = ii+(blockDim.y * blockIdx.y)*stride0;        
                        jjj = jj+(blockDim.x * blockIdx.x)*stride1;                    
                        indo = tz*ci*hi*wi + ig*hi*wi + iii*wi+jjj;
                        if (indo<ci*hi*wi*b)
                            atomicAdd(&(out[indo]), s[indt+ii*(blockDim.x+wk*stride1)+jj]);                                 
                    }
                __syncthreads();
            }
        }
    }
    """,
    "conv2dtranspose",
)
