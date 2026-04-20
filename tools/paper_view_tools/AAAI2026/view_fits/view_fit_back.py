import numpy as np
import sep
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.colors as mcolors

fit_file = '/home/flh/datasets/LAMOST_new/images_1k/20231204_0_image581_I.fit'  # 请替换为你的文件路径
hdul = fits.open(fit_file)
image_data = hdul[0].data

# 修复字节序问题（关键！）
if not image_data.dtype.isnative:
    image_data = image_data.byteswap().newbyteorder()

bkg = sep.Background(image_data)
bkg_image = bkg.back()

zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(bkg_image)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

bkg_fit_file = 'data/tools/paper_view_tools/AAAI2026/view_fits/background_image.fit'
hdu = fits.PrimaryHDU(bkg_image)
hdul_new = fits.HDUList([hdu])
hdul_new.writeto(bkg_fit_file, overwrite=True)

bkg_png_file = 'data/tools/paper_view_tools/AAAI2026/view_fits/background_image.png'
plt.imshow(bkg_image, cmap='gray', origin='lower', interpolation='nearest', norm=norm)
plt.colorbar()
plt.title('Background Image (ZScale)')
plt.savefig(bkg_png_file, dpi=300)
plt.close()

print(f"Background FIT file saved to: {bkg_fit_file}")
print(f"Background PNG saved to: {bkg_png_file}")
