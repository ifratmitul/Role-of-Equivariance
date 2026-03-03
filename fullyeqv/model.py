import torch.nn as nn
from e2cnn import gspaces, nn as enn

class FullyGEquivariantCNN10(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(FullyGEquivariantCNN10, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)

        def make_layer(in_type, out_channels, kernel_size=3, padding=1, pool=False):
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            layers = [
                enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding),
                enn.InnerBatchNorm(out_type),
                enn.ReLU(out_type)
            ]
            if pool:
                layers.append(enn.PointwiseMaxPool(out_type, kernel_size=2))
            return enn.SequentialModule(*layers), out_type

        # Input Field - store this for use in forward method
        self.in_type = enn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

        self.block1, out_type1 = make_layer(self.in_type, 16)
        self.block2, out_type2 = make_layer(out_type1, 32, pool=True)
        self.block3, out_type3 = make_layer(out_type2, 32)
        self.block4, out_type4 = make_layer(out_type3, 64, pool=True)
        self.block5, out_type5 = make_layer(out_type4, 64)
        self.block6, out_type6 = make_layer(out_type5, 64, pool=True)
        self.block7, out_type7 = make_layer(out_type6, 128)
        self.block8, out_type8 = make_layer(out_type7, 128)
        self.block9, out_type9 = make_layer(out_type8, 128)
        self.block10, out_type10 = make_layer(out_type9, 256)

        self.global_pool = enn.PointwiseAdaptiveAvgPool(out_type10, output_size=1)
        self.fc = nn.Linear(256 * self.r2_act.fibergroup.order(), num_classes)

    def forward(self, x):
        # Use the stored input type instead of trying to access block1[0]
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.global_pool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        return self.fc(x)


