import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from adjustText import adjust_text  # Import the adjust_text function


font = FontProperties(family='Times New Roman', style='italic', size=12)

fps = [0.3, 0.8, 4.0, 41.7, 105.8, 65.5, 319.5, 256.2, 51.0, 156.0, 47.3, 50.0, 163.9, 150.0, 212.2, 95.5, 90.0]
miou = [63.1, 78.4, 81.5, 68.0, 68.4, 74.7, 72.7, 73.1, 73.6, 72.6, 75.3, 71.9, 71.5, 72.8, 70.2, 66.3, 70.9]
annotations = ['DeepLab', 'PSPNet', 'DANet', 'ERFNet', 'BiSeNet1', 'BiSeNet2', 'DWRSeg-B50', 'DWRSeg-L50',
               'LBN-AA', 'BiSeNetV2', 'BiSeNetV2-L', 'MSCFNet', 'FasterSeg', 'LETNet', 'RAFNet', 'FPLNet', 'FBSNet']


plt.scatter(fps, miou, marker='o', color='b')

texts = []
for i in range(len(fps)):
    x, y = fps[i], miou[i]
    label = annotations[i]
    if x < 30:
        # Place the text to the right of the point if FPS is less than 30
        texts.append(plt.text(x + 2, y, label, ha='left', va='center', fontproperties=font))
    else:
        # Place the text to the left of the point if FPS is greater than or equal to 30
        texts.append(plt.text(x - 2, y, label, ha='right', va='center', fontproperties=font))


adjust_text(texts, force_text=0.05, expand_text=(1.2, 1.2))

our_fps = [204.7]
our_miou = [74.0]


plt.scatter(our_fps, our_miou, marker='o', color='r')
plt.annotate('MAFNet(Ours)', (our_fps[0], our_miou[0]), xytext=(-20, 10), textcoords='offset points', ha='center', font=font)


plt.axvline(x=30, color='r', linestyle='--')


plt.xlabel('FPS', font=font)
plt.ylabel('mIoU(%)', font=font)


plt.grid(True)


plt.savefig('high_quality_plot1.png', dpi=300, bbox_inches='tight')


plt.show()
