import torch
import torch.nn as nn
import geffnet
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d


sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained) # ! make EfficientNet
        self.dropout_rate=0.5
        if args is not None: 
            self.dropout_rate=args.dropout
        # !
        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(5)
        ])
        print ('output of vec embed from img network {}'.format(self.enet.classifier.in_features))
        self.in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            self.in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(self.in_ch, out_dim) # ! simple classifier
        self.enet.classifier = nn.Identity() # ! pass through, no update

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1) ## flatten ?
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts) # ! takes average output after doing many dropout
        return out


class Resnest_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(Resnest_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        if args is not None: 
            self.dropout_rate=args.dropout
        # !
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.in_ch = self.enet.fc.in_features
        # self.in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(self.in_ch, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Seresnext_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class DualObjectiveNf1Celeb ( Effnet_Melanoma ): 
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(DualObjectiveNf1Celeb, self).__init__(enet_type, out_dim, n_meta_features, n_meta_dim, pretrained)

        self.myfc_celeb = nn.Linear(self.in_ch, len(args.celeb_label)) # ! simple classifier on celeb, there are 40 attributes
        # x = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"

        self.dropouts_celeb = nn.Dropout(0.2) # ! simple dropout for celeb
     
    def forward (self, x, celeb): 
        # @celeb is sent through same network, but eval on different loss. 
        
        # ! forward on skin images
        # print (x.shape)
        x = self.extract(x).squeeze(-1).squeeze(-1) ## flatten
        # print (x.shape)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        # print (out.shape)
        out /= len(self.dropouts) # ! takes average output after doing many dropout

        # ! forward on celeb
        celeb = self.extract(celeb).squeeze(-1).squeeze(-1) ## flatten
        celeb = self.myfc_celeb(self.dropouts_celeb(celeb)) # ! simple dropout

        return out, celeb


class DualObjectiveNf1CelebEvalNf1 ( DualObjectiveNf1Celeb ): 
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(DualObjectiveNf1CelebEvalNf1, self).__init__(enet_type, out_dim, n_meta_features, n_meta_dim, pretrained, args)

    def forward (self, x): 
        # ! can't do forward on both, celeb and skin, not sure how attribution lib works.
        # ! best to just do simple forward on skin 
        # ! forward on skin images
        x = self.extract(x).squeeze(-1).squeeze(-1) ## flatten
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts) # ! takes average output after doing many dropout
        return out



class Celeb24FaceFeat ( Effnet_Melanoma ): 
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(Celeb24FaceFeat, self).__init__(enet_type, out_dim, n_meta_features, n_meta_dim, pretrained)

        self.myfc_celeb = nn.Linear(self.in_ch, len(args.celeb_label)) # ! simple classifier on celeb, there are 40 attributes
        # x = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"

        self.dropouts_celeb = nn.Dropout(0.2) # ! simple dropout for celeb
     
    def forward (self, celeb): 
        # @celeb is sent through same network, but eval on different loss. 

        # ! forward on celeb
        celeb = self.extract(celeb).squeeze(-1).squeeze(-1) ## flatten
        celeb = self.myfc_celeb(self.dropouts_celeb(celeb)) # ! simple dropout

        return celeb


class Celeb24FaceFeatResnest ( Resnest_Melanoma ): 
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(Celeb24FaceFeatResnest, self).__init__(enet_type, out_dim, n_meta_features, n_meta_dim, pretrained)

        self.myfc_celeb = nn.Linear(self.in_ch, len(args.celeb_label)) # ! simple classifier on celeb, there are 40 attributes
        # x = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"

        self.dropouts_celeb = nn.Dropout(0.2) # ! simple dropout for celeb
     
    def forward (self, celeb): 
        # @celeb is sent through same network, but eval on different loss. 

        # ! forward on celeb
        celeb = self.extract(celeb).squeeze(-1).squeeze(-1) ## flatten
        celeb = self.myfc_celeb(self.dropouts_celeb(celeb)) # ! simple dropout

        return celeb

