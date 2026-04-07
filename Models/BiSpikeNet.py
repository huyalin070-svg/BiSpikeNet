class SNNMultiShortcutBlock(nn.Module): 、
    def forward_time(self, x, mem, spike, meta_net):
        residual = x
        out = self.binary_activation(x)
        out1 = self.bn1(self.binary_conv1(out, meta_net))
        out2 = self.bn2(self.binary_conv2(out1, meta_net))
        out3 = self.bn3(self.binary_conv3(out2, meta_net))
        out_sum = out1 + out2 + out3  # multiple shortcuts fusion

        # apply attention if present
        if self.attention is not None:
            out_sum = self.attention(out_sum)

        # integrate & fire
        mem, spk = self.lif(out_sum, mem, spike)

        # apply spike normalization/homeostasis
        if self.use_spikenorm and self.spike_norm is not None:
            spk = self.spike_norm(spk)

        # apply downsample to residual if necessary to match shapes
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.nonlinear(out)

        # optionally freeze batchnorm behavior
        if self.freeze_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()

        return out, mem, spk
