import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# font = FontProperties(family='Times New Roman', style='italic')
font = FontProperties(family='Times New Roman', style='italic',size=12)

fps = [0.3,
        0.8,
        # 4.0,

        41.7,
       105.8,
        # 65.5,
       # 319.5,
       256.2,
       # 51.0,
       # 156.0,
       47.3,
       # 50.0,
       # 163.9,
        150.0,
        212.2,
        95.5,
        90.0,
       ]  # FPS 值
miou = [
        63.1,
        78.4,
       # 81.5,

        68.0,
        68.4,
       # 74.7,
       # 72.7,
       73.1,
       # 73.6,
       # 72.6,
       75.3,
       # 71.9,
        # 71.5,
        72.8,
        70.2,
        66.3,
        70.9,
        ]
annotations = ['DeepLab',
                'PSPNet',
                # 'DANet',
                'ERFNet',

               'BiSeNet1',
               # 'BiSeNet2',
               # 'DWRSeg-B50',
               'DWRSeg-L50',
               # 'LBN-AA',
               # 'BiSeNetV2',
               'BiSeNetV2-L',
               # 'MSCFNet',
               # 'FasterSeg',
               'LETNet',
                'RAFNet',
                'FPLNet',
                'FBSNet',
               ]


plt.scatter(fps, miou, marker='o', color='b')


for i in range(len(fps)):
    plt.annotate(annotations[i], (fps[i], miou[i]), textcoords="offset points", xytext=(0,10), ha='center',font=font)

plt.scatter(4, 81.5, marker='o', color='b')
plt.annotate('DANet', (4, 81.5),textcoords="offset points", xytext=(25,-1), ha='center',font=font)

plt.scatter(319.5, 72.7, marker='o', color='b')
plt.annotate('DWRSeg-B50', (319.5, 72.7),textcoords="offset points", xytext=(5,-20), ha='center',font=font)

plt.scatter(163.9, 71.5, marker='o', color='b')
plt.annotate('FasterSeg', (163.9, 71.5),textcoords="offset points", xytext=(5,-17), ha='center',font=font)

plt.scatter(156, 72.6, marker='o', color='b')
plt.annotate('BiSeNetV2', (156, 72.6),textcoords="offset points", xytext=(32,-3), ha='center',font=font)

plt.scatter(65.5, 74.7, marker='o', color='b')
plt.annotate('BiSeNet2', (65.5, 74.7),textcoords="offset points", xytext=(30,-5), ha='center',font=font)

plt.scatter(51, 73.6, marker='o', color='b')
plt.annotate('LBN-AA', (51, 73.6),textcoords="offset points", xytext=(-30,-5), ha='center',font=font)

plt.scatter(50, 71.9, marker='o', color='b')
plt.annotate('MSCFNet', (50, 71.9),textcoords="offset points", xytext=(-32,-5), ha='center',font=font)

our_fps = [204.7]
our_miou = [74.0]

plt.scatter(our_fps, our_miou, marker='o', color='r')
plt.annotate('MSDSeg(Ours)', (our_fps[0], our_miou[0]),textcoords="offset points", xytext=(-5, 10), ha='center',font=font, color='r')


plt.axvline(x=30, color='r', linestyle='--')

# plt.title('FPS vs mIoU')
plt.xlabel('FPS',font=font)
plt.ylabel('mIoU(%)',font=font)


plt.grid(True)

# plt.savefig('high_quality_plot.png', dpi=300)
plt.savefig('high_quality_plot1.png', dpi=300, bbox_inches='tight')

plt.show()
