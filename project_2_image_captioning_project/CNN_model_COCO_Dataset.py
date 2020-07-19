#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class ConvNet(nn.Module):


    def __init__(self, num_classes = 101):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(64, 128, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(128, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

        )


        self.conv3d = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),

            nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),

            nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

        )


        self.st_classifier = nn.Sequential(
            nn.Linear(9216, 4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input):

        x = self.features(input)
        x = self.conv3d(x)

        x = x.view(-1, 9216)
        x = self.st_classifier(x)
        return x

