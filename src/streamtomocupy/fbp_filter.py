import cupy as cp

from streamtomocupy import cfunc_filter
from streamtomocupy import cfunc_filterfp16

class FBPFilter():
    def __init__(self, n, ntheta, nz, dtype):
        if dtype == 'float16':
            self.fslv = cfunc_filterfp16.cfunc_filter(ntheta, nz, n)
        else:
            self.fslv = cfunc_filter.cfunc_filter(ntheta, nz, n)
        self.n = n

    def filter(self, data, w, stream):
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(data)
        w = cp.ascontiguousarray(w.view('float32').astype(data.dtype))
        self.fslv.filter(data.data.ptr, w.data.ptr, stream.ptr)

    def calc_filter(self, filter):
        d = 0.5
        t = cp.arange(0, self.n/2+1)/self.n

        if filter == 'none':
            wfa = self.n*0.5+t*0
            wfa[0] *= 2  # fixed later
        elif filter == 'ramp':
            wfa = self.n*0.5*self._wint(12, t)
        elif filter == 'shepp':
            wfa = self.n*0.5*self._wint(12, t)*cp.sinc(t/(2*d))*(t/d <= 2)
        elif filter == 'cosine':
            wfa = self.n*0.5*self._wint(12, t)*cp.cos(cp.pi*t/(2*d))*(t/d <= 1)
        elif filter == 'cosine2':
            wfa = self.n*0.5*self._wint(12, t) * \
                (cp.cos(cp.pi*t/(2*d)))**2*(t/d <= 1)
        elif filter == 'hamming':
            wfa = self.n*0.5 * \
                self._wint(12, t)*(.54 + .46 * cp.cos(cp.pi*t/d))*(t/d <= 1)
        elif filter == 'hann':
            wfa = self.n*0.5*self._wint(12, t) * \
                (1+cp.cos(cp.pi*t/d)) / 2.0*(t/d <= 1)
        elif filter == 'parzen':
            wfa = self.n*0.5*self._wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)

        wfa = 2*wfa*(wfa >= 0)
        wfa[0] *= 2
        wfa = wfa.astype('float32')
        return wfa

    def _wint(self, n, t):

        N = len(t)
        s = cp.linspace(1e-40, 1, n)
        # Inverse vandermonde matrix
        tmp1 = cp.arange(n)
        tmp2 = cp.arange(1, n+2)
        iv = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))
        u = cp.diff(cp.exp(cp.outer(tmp2, cp.log(s)))*cp.tile(1.0 /
                    tmp2[..., cp.newaxis], [1, n]))  # integration over short intervals
        W1 = cp.matmul(iv, u[1:n+1, :])  # x*pn(x) term
        W2 = cp.matmul(iv, u[0:n, :])  # const*pn(x) term

        # Compensate for overlapping short intervals
        tmp1 = cp.arange(1, n)
        tmp2 = (n-1)*cp.ones((N-2*(n-1)-1))
        tmp3 = cp.arange(n-1, 0, -1)
        p = 1/cp.concatenate((tmp1, tmp2, tmp3))
        w = cp.zeros(N)
        for j in range(N-n+1):
            # Change coordinates, and constant and linear parts
            W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2

            for k in range(n-1):
                w[j:j+n] = w[j:j+n] + p[j+k]*W[:, k]

        wn = w
        wn[-40:] = (w[-40])/(N-40)*cp.arange(N-40, N)
        return wn
