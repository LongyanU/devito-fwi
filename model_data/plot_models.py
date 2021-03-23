import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib.pyplot import gca
from pylab import rcParams
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import pickle

#fp = open('visu/cmap_cm.pkl', 'rb')
my_cmap_cm = 'jet'#'bwr'#pickle.load(fp)#
#fp.close()

DH = 30.0
NX = 300
NY = 106
NPML = 0
FREE_SURFACE = 0
vpmin=800.0

xtick = [0., 2., 4., 6., 8.]
ytick = [0., 1., 2., 3., ]

FSize = 16
font = {'color':  'black',
		'weight': 'normal',
		'size': FSize}
mpl.rc('xtick', labelsize=FSize) 
mpl.rc('ytick', labelsize=FSize) 
rcParams['figure.figsize'] = 10, 4

bathy_pos = 7
apply_bathy = False#True#
vp_range = [1.5e3, 5e3]#[-2e-18, 2e-18]#
vp_ticks = [1.5e3, 2e3, 2.5e3, 3.e3, 3.5e3, 4.e3, 4.5e3, 5.e3]
vp_ticklabels = [1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
vs_range = [1e3, 3e3]
rho_range = [-5e-8, 5e-8]#[1e3, 3e3]#

show_vp = True
show_rho = False
show_vs = False
do_annotate = False#True#
annotate_xy = [(4.2, 1.3), (5.5, 1.5)]#(2.0, 2.6), , (2.0, 2.0)
annotate_xytext = [(3.7, 1.0),  (5.0,1.2)]#(1.5, 2.3),, (1.5, 1.7)
do_retangle = False#True#
rect_xy = [(3.0, 1.8)]
rect_width = [4.0]
rect_height = [1.3]

vp_file_name = "./SMARMN/vp.smooth3"#
rho_file_name = ""
vs_file_name = ""
fig_file_name = "./SMARMN/vp.smooth3.eps"#
save_fig = True
fig_nrow = 1
fig_ncol = 1
fig_format = 'eps'

if show_vp:
	f = open(vp_file_name)
	data_type = np.dtype('float32').newbyteorder('<')
	vp = np.fromfile(f, dtype=data_type)
	vp = vp.reshape(NX,NY)
	vp = np.transpose(vp)
	vp = np.flipud(vp)

	if FREE_SURFACE==1:
		vp = vp[:NY-NPML, NPML:NX-NPML]
	elif FREE_SURFACE==0:
		vp = vp[NPML:NY-NPML, NPML:NX-NPML]
	else:
		pass
	print(vp.shape)
	if apply_bathy:
		vp[-bathy_pos:, :] = 0
	NY, NX = vp.shape
	print("vp max value: ", np.max(np.abs(vp)))

if show_vs:
	f = open(vs_file_name)
	data_type = np.dtype('float32').newbyteorder('<')
	vs = np.fromfile(f, dtype=data_type)
	vs = vs.reshape(NX,NY)
	vs = np.transpose(vs)
	vs = np.flipud(vs)
	print("vs max value: ", np.max(np.abs(vs)))
	if FREE_SURFACE==1:
		vs = vs[:NY-NPML, NPML:NX-NPML]
	elif FREE_SURFACE==0:
		vs = vs[NPML:NY-NPML, NPML:NX-NPML]
	else:
		pass
	print(vs.shape)
	NY, NX = vs.shape

if show_rho:
	f = open(rho_file_name)
	data_type = np.dtype('float32').newbyteorder('<')
	rho = np.fromfile(f, dtype=data_type)
	rho = rho.reshape(NX,NY)
	rho = np.transpose(rho)
	rho = np.flipud(rho)
	print("rho max value: ", np.max(np.abs(rho)))
	if FREE_SURFACE==1:
		rho = rho[:NY-NPML, NPML:NX-NPML]
	elif FREE_SURFACE==0:
		rho = rho[NPML:NY-NPML, NPML:NX-NPML]
	else:
		pass
	print(rho.shape)
	NY, NX = rho.shape

x = np.arange(0.0, DH*NX, DH)
y = np.arange(0.0, DH*NY, DH)
x = np.divide(x,1000.0);
y = np.divide(y,1000.0);

def do_plot(nrow, ncol, n, model, cm, an, title, vp_range, cbar_ticks=None, cbar_ticklabels=None):
	vpmin, vpmax = vp_range[0], vp_range[1]
	ax=plt.subplot(nrow, ncol, n)
	ax.set_xticks(xtick)
	ax.set_yticks(ytick)

	#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	## for Palatino and other serif fonts use:
	#rc('font',**{'family':'serif','serif':['Palatino']})
	#plt.rc('text', usetex=True)
	rc('text', usetex=True)

	# plt.pcolor(x, y, vp, cmap=cm, vmin=vpmin)
	im = ax.imshow(model, cmap=cm, interpolation='none',
			extent=[0.0,NX*DH/1000.0,0.0,NY*DH/1000.0], 
			vmin=vpmin, vmax=vpmax)
	# a = gca()

	ax.set_xticklabels(ax.get_xticks(), font)
	ax.set_yticklabels(ax.get_yticks(), font)
	plt.axis('scaled')
	plt.ylabel('Depth [km]', fontdict=font)

	plt.xlabel('Distance [km]', fontdict=font)
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top') 
	plt.gca().invert_yaxis()

	if do_annotate:
		if len(annotate_xy) != len(annotate_xytext):
			raise Exception("annotation xy doesn't match xytext")
		for i in range(len(annotate_xy)):
			ax.annotate('', xy=annotate_xy[i], xytext=annotate_xytext[i], 
				arrowprops=dict(shrink=.2, color='white'))

	if do_retangle:
		for i in range(len(rect_xy)):
			rect = patches.Rectangle(rect_xy[i], rect_width[i], rect_height[i], 
						ls='--', lw=2, ec='w', fc='None')
			ax.add_patch(rect)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="1%", pad=0.05)
	#cbar=plt.colorbar(im, aspect=20, pad=0.02, shrink=1)
	cbar = plt.colorbar(im, cax=cax, ticks=cbar_ticks)
	cbar.set_label(title, fontdict=font, labelpad=5)
	cbar.ax.set_yticklabels(cbar_ticklabels)
	plt.text(0.1, 0.32, an, fontdict=font, color='white')
	plt.tight_layout()


plt.close('all')
plt.figure()

idx = 0
if show_vp:
	idx += 1
	do_plot(fig_nrow, fig_ncol, idx, vp, my_cmap_cm, '', r"$\rm{V_p [km/s]}$", vp_range, vp_ticks, vp_ticklabels)
if show_vs:
	idx += 1
	do_plot(fig_nrow, fig_ncol, idx, vs, my_cmap_cm, '', r"$\rm{V_s [km/s]}$", vs_range)
if show_rho:
	idx += 1
	do_plot(fig_nrow, fig_ncol, idx, rho, my_cmap_cm, '', r"$\rm{\rho [kg/m^3]}$", rho_range)
if save_fig:
	if fig_format=='png':
		plt.savefig(fig_file_name, format=fig_format, dpi=300, bbox_inches='tight')#
	else:
		plt.savefig(fig_file_name, format=fig_format)#, bbox_inches='tight', pad_inches=.25
plt.show()