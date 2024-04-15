import numpy
import os
import cv2
import torch
from torch import nn, optim
from torchvision import models
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import yaml
import copy

class CNN_LSTM:
    def __init__(self):
        self.cnn_model = None
        self.lstm_model = None
        self.task_class_mapping = None
        self.sequence = None

    def loadModel(self, modelFolder, modelName=None):
        self.loadConfig(modelFolder)
        self.loadCNNModel(modelFolder)
        self.loadLSTMModel(modelFolder)

    def loadConfig(self,modelFolder):
        with open(os.path.join(modelFolder,"config.yaml"),"r") as f:
            config = yaml.safe_load(f)
        self.task_class_mapping = config["class_mapping"]
        self.sequence_length = config["sequence_length"]
        self.num_features = config["num_features"]
        self.sequence = numpy.zeros((self.sequence_length, self.num_features))
        self.num_classes = len([key for key in self.task_class_mapping])
        self.device = config["device"]

    def loadCNNModel(self,modelFolder):
        self.cnn_model = ResNet_FeatureExtractor(self.num_features, self.num_classes)
        res_ckpt = torch.load(os.path.join(modelFolder, "resnet.pth"), map_location="cpu")
        self.cnn_model.load_state_dict(res_ckpt["model"], strict=True)
        self.transforms = self.cnn_model.transforms
        try:
            self.cnn_model.cuda(self.device)
        except AttributeError:
            pass
        return self.cnn_model

    def loadLSTMModel(self,modelFolder):
        self.lstm_model = WorkflowLSTM(num_features=self.num_features,num_classes = self.num_classes)
        checkpoint = torch.load(os.path.join(modelFolder,"lstm.pth"))
        self.lstm_model.load_state_dict(checkpoint["model"],strict=True)
        try:
            self.lstm_model.cuda(self.device)
        except AttributeError:
            pass
        return self.lstm_model

    def predict(self,image):
        self.cnn_model.eval()
        self.lstm_model.eval()
        with torch.no_grad():
            img_tensor = self.transforms(image.resize((224,224)))
            image = torch.from_numpy(numpy.array([img_tensor])).cuda(self.device)
            preds = self.cnn_model.forward(image)
            pred = preds.cpu().detach().numpy()
            self.sequence = numpy.concatenate((self.sequence[:-1,],pred),axis=0)
            expanded_sequence = numpy.expand_dims(self.sequence,axis=0).astype(float)
            taskPrediction = self.lstm_model(torch.from_numpy(expanded_sequence).float().cuda(self.device))
            taskPrediction = taskPrediction.cpu().numpy()
            class_num = numpy.argmax(taskPrediction[0][-1])
            networkOutput = str(self.task_class_mapping[class_num]) + str(taskPrediction)
            return networkOutput

    def createCNNModel(self,num_input_features,num_classes):
        self.cnn_model = ResNet_FeatureExtractor(num_input_features, num_classes)
        self.num_classes = num_classes
        self.num_features = num_input_features
        return self.cnn_model

    def createLSTMModel(self,num_features,num_classes):
        self.lstm_model = WorkflowLSTM(num_features,num_classes)
        return self.lstm_model

class WorkflowLSTM(nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.hidden_size, num_layers=2, batch_first = True,bidirectional=False, dropout=0.2)
        self.linear_1 = nn.Linear(128,64)
        self.relu1 = nn.ReLU()
        self.droput1 = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(64, num_classes)
        self.relu2 = nn.ReLU()
        self.droput2 = nn.Dropout(0.2)
        self.linear_3 = nn.Linear(num_classes, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self,sequence):
        sequence = sequence
        #h_0 = torch.autograd.Variable(torch.zeros(1, sequence.size(0), self.hidden_size)).cuda("cuda")  # hidden state
        #c_0 = torch.autograd.Variable(torch.zeros(1, sequence.size(0), self.hidden_size)).cuda("cuda")
        x, (hn,cn) = self.lstm(sequence)
        last_time_step = x[:, -1, :]
        #x = self.relu1(x)
        x = self.linear_1(x)
        x = self.relu1(x)
        x = self.linear_2(x)

        scores = F.log_softmax(x,dim=1)
        '''x = self.relu1(x)
        x = self.droput1(x)
        x = self.linear_2(x)
        x = self.relu2(x)
        x = self.droput2(x)
        x = self.linear_3(x)'''
        #x = self.softmax(x)
        return x

class TCN(nn.Module):
    def __init__(self,input_size,output_size,num_channels,kernel_size=2,dropout=0.2):
        super(TCN,self).__init__()
        self.tcn = TemporalConvNet(input_size,num_channels,kernel_size=kernel_size,dropout=dropout)
        self.class_layer = nn.Linear(input_size,output_size)
        '''self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(output_size, output_size)'''
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x = self.tcn(x.transpose(1,2))
        #last_time_step = x[:, -1, :]
        x = self.class_layer(x.transpose(1,2))
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.linear_2(x)
        #x = self.softmax(x)
        last_time_step = x[:, -1, :]
        return last_time_step

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MultiStageModel(nn.Module):
    def __init__(self, stages = 4, layers = 10, feature_maps = 64, feature_dimension = 2048,out_features = 10, causal_conv = True):
        self.num_stages = stages #hparams.mstcn_stages  # 4 #2
        self.num_layers = layers #hparams.mstcn_layers  # 10  #5
        self.num_f_maps = feature_maps #hparams.mstcn_f_maps  # 64 #64
        self.dim = feature_dimension #hparams.mstcn_f_dim  #2048 # 2048
        self.num_classes = out_features #hparams.out_features  # 7
        self.causal_conv = causal_conv #hparams.mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_output = []
        for i in range(x.size(0)):
            out_classes = self.stage1(x[i].unsqueeze(0).transpose(2,1))
            outputs_classes = out_classes.unsqueeze(0)
            for s in self.stages:
                out_classes = s(F.softmax(out_classes, dim=1))
                outputs_classes = torch.cat(
                    (outputs_classes, out_classes.unsqueeze(0)), dim=0)
            #outputs_classes = self.softmax(outputs_classes)
            outputs_classes=outputs_classes.transpose(2,3)
            batch_output.append(outputs_classes)
        return torch.stack(batch_output).squeeze(2)

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(
            title='mstcn reg specific args options')
        mstcn_reg_model_specific_args.add_argument("--mstcn_stages",
                                                   default=4,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_layers",
                                                   default=10,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_maps",
                                                   default=64,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_dim",
                                                   default=2048,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_causal_conv",
                                                   action='store_true')
        return parser


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedSmoothLayer(nn.Module):
    def __init__(self, causal_conv=True):
        super(DilatedSmoothLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation1 = 1
        self.dilation2 = 5
        self.kernel_size = 5
        if self.causal_conv:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2 * 2,
                                           dilation=self.dilation2)

        else:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2,
                                           dilation=self.dilation2)
        self.conv_1x1 = nn.Conv1d(7, 7, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x1 = self.conv_dilated2(x1[:, :, :-4])
        out = F.relu(x1)
        if self.causal_conv:
            out = out[:, :, :-((self.dilation2 * 2) * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class ResNet_FeatureExtractor(nn.Module):
    def __init__(self,num_output_features,num_classes,multitask=False,num_tools=0,return_head=True):
        super(ResNet_FeatureExtractor,self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.resnet = models.resnet50(weights=weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.linear1 = nn.Linear(num_features,num_output_features)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_head = nn.Linear(num_output_features,num_classes)
        if multitask:
            self.tool_head = nn.Linear(num_output_features,num_tools)
        #self.softmax = nn.Softmax(dim=1)
        self.return_head = return_head
        self.multitask = multitask

    def forward(self,x):
        x = self.resnet(x)
        x = self.linear1(x)
        x = self.sig(x)
        if self.return_head:
            if self.multitask:
                task = self.output_head(x)
                tool = self.tool_head(x)
                return task,tool
            else:
                x = self.output_head(x)
            #x = self.softmax(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x