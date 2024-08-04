import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from model_CNN import *
from model_transformer import *
import configs
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
config = CONFIGS["ViT-B_16"]
num_classes=10
model = VisionTransformer(config, zero_head=True, num_classes=num_classes)
# model = CNN(
#     in_channels=2, out_channels=64, out_channels_new=32, num_classes=10
# )
model.load_state_dict(
    torch.load(
        "/DATA/nfsshare/Amartya/EMNLP-WACV/communication_journal/checkpoint_transformer/communication_checkpoint.bin"
    )
)
model.to(device)
model.eval()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


Train_Dataset = Communicationdataset(
    root="/DATA/nfsshare/Amartya/EMNLP-WACV/communication_journal",
)
train_size = int(0.7 * len(Train_Dataset))
valid_size = len(Train_Dataset) - train_size
trainset, testset = torch.utils.data.random_split(
    Train_Dataset, [train_size, valid_size]
)
test_sampler = SequentialSampler(testset)
test_loader = DataLoader(
    testset,
    sampler=test_sampler,
    batch_size=1,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)
labels = {
    0: "8PSK",
    1: "AM-DSB",
    2: "AM-SSB",
    3: "BPSK",
    4: "CPFSK",
    5: "GFSK",
    6: "PAM4",
    7: "QAM16",
    8: "QAM64",
    9: "QPSK",
    10: "WBFM"
}

lab = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM"]

all_preds, all_label = [], []
for step, batch in enumerate(tqdm(test_loader)):
    batch = tuple(t.to(device) for t in batch)
    x, y = batch
    x = x.float()
    with torch.no_grad():
        logits = model(x)[0]
        preds = torch.argmax(logits, dim=-1)
    if len(all_preds) == 0:
        all_preds.append(preds.detach().cpu().numpy())
        all_label.append(y.detach().cpu().numpy())
    else:
        all_preds[0] = np.append(
            all_preds[0], preds.detach().cpu().numpy(), axis=0
        )
        all_label[0] = np.append(
            all_label[0], y.detach().cpu().numpy(), axis=0
        )
        
all_preds, all_label = all_preds[0], all_label[0]
accuracy = simple_accuracy(all_preds, all_label)
print(accuracy)

cm = confusion_matrix(all_label, all_preds, labels=np.arange(len(labels)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lab)
disp.plot()
plt.show()
# %%
