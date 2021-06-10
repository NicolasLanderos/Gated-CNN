import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import numpy as np

def Plot_Experiments(Nexp,Weights_id,Dataset_name,Net_name, figsize, Qbits, xlabel, E_acc, E_loss, NQacc, NQloss , save_fig=False):
	fig, axs = plt.subplots(1,2,figsize = figsize)
	fig.suptitle(Net_name+' Experiment '+ str(Nexp), fontsize=20)
	
	axs[0].set_xticks(Qbits)
	axs[0].set_ylim([0, 1])
	axs[0].set_xlabel(xlabel)
	axs[0].set_ylabel('Accuracy',fontsize=14)
	axs[0].axhline(y=NQacc, color='r', linestyle='-')
	axs[0].plot(Qbits, E_acc, color = 'b', linestyle='-')
	axs[0].legend(('Original Model','Quantized Model'))
	
	
	axs[1].set_xticks(Qbits)
	axs[1].set_ylim([0, int(np.floor(np.max(E_loss)) + 1)])
	axs[1].set_xlabel(xlabel)
	axs[1].set_ylabel('Loss',fontsize=14)
	axs[1].axhline(y=NQloss, color='r', linestyle='-')
	axs[1].plot(Qbits, E_loss, color = 'b', linestyle='-')
	axs[1].legend(('Original Model','Quantized Model'))
	
	if save_fig:
		fig.savefig('Figures/Quantization Experiments/'+str(Net_name)+'/'+str(Dataset_name)+'/'+str(Weights_id)+'/Experiment'+str(Nexp)+'.png')


def Plot_4Experiments(Net_name, Qbits, Accs, Losses, OrigAcc, OrigLoss , xlabels, figsize, save_fig=False, fig_dir = None):
	if Accs and Losses:
		fig, axs = plt.subplots(4,2,figsize = figsize)
	else:
		fig, axs = plt.subplots(4,1,figsize = figsize)
		axs = axs.reshape([4,1])
	fig.set_facecolor('0.75')
	fig.suptitle(Net_name+' Quantization Results ', fontsize=20)
	for row in range(4):
		if Accs and Losses:
			for col in range(2):
				if col == 0:
					axs[row,col].set_ylim([0, 1])
					axs[row,col].set_ylabel('Accuracy',fontsize=14)
					axs[row,col].axhline(y=OrigAcc, color='r', linestyle='-')
					axs[row,col].plot(Qbits[row], Accs[row], color = 'b', linestyle='-')
				else:
					axs[row,col].set_ylim([0, 1.2*np.max(Losses[row]) ])
					axs[row,col].set_ylabel('Loss',fontsize=14)
					axs[row,col].axhline(y=OrigLoss, color='r', linestyle='-')
					axs[row,col].plot(Qbits[row], Losses[row], color = 'b', linestyle='-')
				axs[row,col].set_xticks(Qbits[row])	
				axs[row,col].set_xlabel(xlabels[row],fontsize=13)
				axs[row,col].legend(('Original Model','Quantized Model'))
		else:
			axs[row,0].set_ylim([0, 1.2*np.max(Losses[row]) ])
			axs[row,0].set_ylabel('Loss',fontsize=14)
			axs[row,0].axhline(y=OrigLoss, color='r', linestyle='-')
			axs[row,0].plot(Qbits[row], Losses[row], color = 'b', linestyle='-')
			axs[row,0].set_xticks(Qbits[row])	
			axs[row,0].set_xlabel(xlabels[row],fontsize=13)
			axs[row,0].legend(('Original Model','Quantized Model'))

	fig.tight_layout()
	fig.subplots_adjust(top=0.95)
	if save_fig:
		fig.savefig(fig_dir + 'Quantization Results.png')




def Plot_Regression_Experiments(Nexp,Net_name,figsize,Qbits,xlabel,E_loss,NQloss,fig_dir=None,save_fig=False):
	plt.figure(figsize=figsize)
	plt.title(Net_name+' Experiment '+ str(Nexp), fontsize=20)
	plt.xticks(Qbits)
	plt.xlabel(xlabel)
	plt.ylabel('Loss',fontsize=14)
	plt.axhline(y=NQloss, color='r', linestyle='-')
	plt.plot(Qbits, E_loss, color = 'b', linestyle='-')
	plt.legend(('Original Model','Quantized Model'))
	if save_fig:
		plt.savefig(fig_dir +'/Experiment'+str(Nexp)+'.png')


def plot_duty(duty, ciclos, title, fig_dir, bits_dist, low, high, mode = 'Duty', figsize = (15,15), threshold = None, save_fig = False, sections_boundaries = None):
	word_size = np.sum(bits_dist)
	rows = high - low
	data = duty/ciclos
	if mode == 'Duty':
		mx = 0.5
		mn = 0
		data[data > 0.5] = 1 - data[data > 0.5]
	elif mode == 'Detect Higher Than':
		mx = 1
		mn = 0
		data[data < threshold] = 0
	elif mode == 'Detect Lower Than':
		mx = 1
		mn = 0
		data[data > threshold] = 1
	data = np.reshape(data[low*word_size:high*word_size],(rows,word_size))
	if len(bits_dist) == 3:
		fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= figsize, gridspec_kw={'width_ratios': bits_dist})
		fig.suptitle(title, fontsize=20)

		ax1.set_title('Sign bit')
		im1 = ax1.imshow(data[...,0:1], aspect='auto',cmap='YlOrRd',vmin=mn, vmax=mx)
		ax1.ticklabel_format(style='plain')
		ax1.set_xticks([0])

		ax2.set_title('Int bits')
		im2 = ax2.imshow(data[...,1:bits_dist[0]+bits_dist[1]], aspect='auto',cmap='YlOrRd',vmin=mn, vmax=mx)
		ax2.set_xticks(np.arange(0,bits_dist[1]))
		ax2.set_xticklabels(np.arange(0,bits_dist[1])+bits_dist[0])
		ax2.yaxis.set_visible(False)

		ax3.set_title('Frac bits')
		im3 = ax3.imshow(data[...,bits_dist[0]+bits_dist[1]:word_size], aspect='auto',cmap='YlOrRd',vmin=mn, vmax=mx)
		ax3.set_xticks(np.arange(0,bits_dist[2]))
		ax3.set_xticklabels(np.arange(0,bits_dist[2])+bits_dist[0]+bits_dist[1])
		ax3.yaxis.set_visible(False)
		divider3 = make_axes_locatable(ax3)
		cax3 = divider3.append_axes("right", size="20%", pad=0.05)
		cbar3 = plt.colorbar(im3, cax=cax3, format="%.2f")
		ax3.yaxis.set_visible(False)

		if sections_boundaries:
			ax1.hlines(sections_boundaries, *ax1.get_xlim())
			ax2.hlines(sections_boundaries, *ax2.get_xlim())
			ax3.hlines(sections_boundaries, *ax3.get_xlim())

	
	elif len(bits_dist) ==2:
		fig, (ax1, ax2) = plt.subplots(1,2,figsize= figsize, gridspec_kw={'width_ratios': bits_dist})
		fig.suptitle(title, fontsize=20)

		ax1.set_title('Sign bit')
		im1 = ax1.imshow(data[...,0:1], aspect='auto',cmap='YlOrRd',vmin=mn, vmax=mx)
		ax1.ticklabel_format(style='plain')
		ax1.set_xticks([0])

		ax2.set_title('Frac bits')
		im2 = ax2.imshow(data[...,1:word_size], aspect='auto',cmap='YlOrRd',vmin=mn, vmax=mx)
		ax2.set_xticks(np.arange(0,bits_dist[1]))
		ax2.set_xticklabels(np.arange(0,bits_dist[1])+1)
		ax2.yaxis.set_visible(False)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes("right", size="20%", pad=0.05)
		cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")
		ax2.yaxis.set_visible(False)

		if sections_boundaries:
			ax1.hlines(sections_boundaries, *ax1.get_xlim())
			ax2.hlines(sections_boundaries, *ax2.get_xlim())

	if save_fig:
		fig.savefig(fig_dir + '.png')


#DataDicts Structure: Errortype->Buffer->Section->ErrorNumber
def Plot_Errors(DataDicts, Sections, keys1, keys2, OgValue, figsize, xlabel, Scaled_max_value=False, save_fig=False, fig_dir = '', BoxPlot = False):
	cols = len(DataDicts)

	fig = plt.figure(constrained_layout=True, figsize=figsize)
	subfigs = fig.subfigures(2, 1, height_ratios = [len(Sections[0]),len(Sections[1])], hspace = 0.05)

	for subfig_index, subfig in enumerate(subfigs):
		subfig.set_facecolor('0.75')
		subfig.suptitle('Buffer ' + str(subfig_index+1), fontsize = 22)
		rows = len(Sections[subfig_index])
		axs = subfig.subplots(rows, cols)
		axs = axs.reshape([rows,cols])
		for row in range(rows):
			for col in range(cols):
				Dict = DataDicts[keys1[col]][keys2[subfig_index]][Sections[subfig_index][row]]
				axs[row,col].set_title(Sections[subfig_index][row]+', Error Type: static 1 in bit: '+str(col+1), fontsize=14)
				axs[row,col].set_xlabel(xlabel, fontsize = 13)
				axs[row,col].set_ylabel('Loss',fontsize=13)

				if BoxPlot:
					labels, data = Dict.keys(), Dict.values()        
					axs[row,col].boxplot(data)
					axs[row,col].set_xticks(range(1, len(labels) + 1))
					axs[row,col].set_xticklabels(labels)
				else:
					Dict = {str(k):   Dict[k]   for k in Dict}
					Dict = {k: np.mean(Dict[k]) for k in Dict}
					axs[row,col].bar(Dict.keys(), Dict.values(), 0.2)
					axs[row,col].axhline(y=OgValue,linewidth=1, color='r')
				
				if Scaled_max_value:
					axs[row,col].set_ylim([0, 1.2*Scaled_max_value])
	
	plt.show()
	if save_fig:
		fig.savefig(fig_dir + '.png')
