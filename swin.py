
imagefolderpath = 'IMAGE'

swin = models.swin_t()

swin.head = nn.Linear(768, 10)

allimagenames = os.listdir(imagefolderpath)
allimagetensor = []
transformation1 = transforms.ToTensor()
transformation2 = transforms.Resize((256,256))

for name in allimagenames:
    
    img = Image.open(imagefolderpath + '/' + name)
    img = transformation1(img)
    img = transformation2(img)

    
    allimagetensor.append(img)
print(len(allimagetensor))

allimagetensor2 = torch.stack(allimagetensor)
outputs = []
for i in tqdm(allimagetensor2):
  i = i[None, :, :, :]
  i = i.to(device)
  output = swin(i)
  outputs.append(output.detach())
  
  
output10 = torch.stack(outputs)
print(output10.size())

targets = torch.Tensor(output10).clone().detach()
print(targets)



