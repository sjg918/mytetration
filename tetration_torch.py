
import numpy as np
import cv2
import time
import torch
import torch.nn as nn

# refer : https://www.youtube.com/watch?v=pspWIwMFgws
# refer : https://github.com/DMTPARK/mytetration


def generate_complex_tensor(x0, y0, eps, n):
    x = np.linspace(x0 - eps, x0 + eps, n, dtype=np.float64)
    y = np.linspace(y0 - eps, y0 + eps, n, dtype=np.float64)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    c = c.astype(np.complex128)
    complex_tensor = torch.from_numpy(c)
    return complex_tensor


class TorchContainer(nn.Module):

    def __init__(self, escape_radius):
        super().__init__()
        self.escape_radius = escape_radius
        foo = nn.Embedding(2, 2)
        self.foo = foo.weight

    def forward(self, complex_tensor, maxiter):
        #print(complex_tensor.shape, complex_tensor.dtype)
        complex_tensor = torch.complex(complex_tensor[..., 0], complex_tensor[..., 1]).to(torch.complex128)
        divergence_map = torch.zeros(complex_tensor.shape, dtype=torch.uint8, device=self.foo.device)
        z = complex_tensor.clone()
        for _ in range(maxiter):
            z = torch.pow(complex_tensor, z)
            mask = torch.abs(z) > escape_radius
            divergence_map[mask] = 255
            complex_tensor[mask] = torch.nan
            continue
        return divergence_map


def mainfunc(x0_list, y0_list, eps_list, n, divergence_max_iter, max_step, focus_step, escape_radius, gpunum):
    tc = TorchContainer(escape_radius=escape_radius).to(device='cuda:0')
    tc = nn.DataParallel(tc, device_ids=(0, 1, 2, 3))
    tc.eval()

    escape_flag = False
    timelist = []
    with torch.no_grad():
        for i in range(max_step):
            print(i)
            x0 = x0_list[i]
            y0 = y0_list[i]
            eps = eps_list[i]
            max_iter = divergence_max_iter[i]
            
            complex_tensor = generate_complex_tensor(x0, y0, eps, n)

            torch.cuda.synchronize()
            starttime = time.time()
            complex_tensor = complex_tensor.view(gpunum, n // gpunum, n).to(device='cuda:0')
            divergence_map = tc(complex_tensor, max_iter)
            torch.cuda.synchronize()
            if i > 2:
                timelist.append(time.time() - starttime)

            divergence_map = divergence_map.view(n, n).cpu().numpy()
            cv2.imwrite('./outputs/' + str(i).zfill(6) + '.png', divergence_map)

            dx = cv2.Sobel(divergence_map, -1, dx=1, dy=0, ksize=3).astype(np.float32)
            dy = cv2.Sobel(divergence_map, -1, dx=1, dy=0, ksize=3).astype(np.float32)
            dd = (np.abs(dx) + np.abs(dy)) / 2
            
            dmask = dd > 50
            dd4 = dd[n//2-n//8:n//2+n//8, n//2-n//8:n//2+n//8]
            dd4mask = dd4 > 50
            
            # if i > focus_step and np.sum(dd4mask) < n*n*0.005 or np.sum(dmask) < n*n*0.005:
            #     x0_list = x0_list[:i+1]
            #     y0_list = y0_list[:i+1]

            #     complex_np = torch.view_as_real(complex_tensor).cpu().view(n, n, 2).numpy()
            #     complex_real = complex_np[:, :, 0]
            #     complex_imag = complex_np[:, :, 1]

            #     complex_real = complex_real[dmask].reshape(-1)
            #     complex_imag = complex_imag[dmask].reshape(-1)
            #     ri = np.random.randint(0, len(complex_real))
            #     x1 = complex_real[ri]
            #     y1 = complex_imag[ri]
            #     new_x0 = np.linspace(x0, x1, focus_step//4, dtype=np.float64)
            #     new_y0 = np.linspace(y0, y1, focus_step//4, dtype=np.float64)
            #     x0_list = np.concatenate((x0_list, new_x0[1:]), axis=-1)
            #     y0_list = np.concatenate((y0_list, new_y0[1:]), axis=-1)
            #     continue

            if len(x0_list) - 1 == i:
                complex_np = torch.view_as_real(complex_tensor).cpu().view(n, n, 2).numpy()
                complex_real = complex_np[n//2-n//8:n//2+n//8, n//2-n//8:n//2+n//8, 0]
                complex_imag = complex_np[n//2-n//8:n//2+n//8, n//2-n//8:n//2+n//8, 1]

                complex_real = complex_real[dd4mask].reshape(-1)
                complex_imag = complex_imag[dd4mask].reshape(-1)
                ri = np.random.randint(0, len(complex_real))
                x1 = complex_real[ri]
                y1 = complex_imag[ri]
                new_x0 = np.linspace(x0, x1, focus_step, dtype=np.float64)
                new_y0 = np.linspace(y0, y1, focus_step, dtype=np.float64)
                x0_list = np.concatenate((x0_list, new_x0), axis=-1)
                y0_list = np.concatenate((y0_list, new_y0), axis=-1)
                continue
            continue

    print(sum(timelist) / len(timelist))
    # end

if __name__ == '__main__':
    focus_step = 16
    x0_list = np.linspace(0., -4.2, focus_step, dtype=np.float64)
    y0_list = np.linspace(0., 0., focus_step, dtype=np.float64)
    
    max_step = 160
    eps_step_scale = 1.2
    eps_list = [1e1 / (pow(eps_step_scale,i)) for i in range(1,max_step+1)]
    divergence_max_iter = [500 + i * 5 for i in range(max_step)]
    escape_radius = 1e+10
    n = 1024
    gpunum = 1

    mainfunc(x0_list, y0_list, eps_list, n, divergence_max_iter, max_step, focus_step, escape_radius, gpunum)
