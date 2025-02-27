import numpy as np


class Convolution:

    def __init__(self,
                 input_channels: int,
                 output_channels:int,
                 kernel_size: int = 2,
                 padding: int = 1,
                 stride: int = 1
                 ):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.kernel_w = np.random.randn(
            self.output_channels, self.input_channels, self.kernel_size,
            self.kernel_size
        ) * np.sqrt(2 / self.input_channels)
        self.kernel_b: np.ndarray = np.zeros(
            output_channels
        )

        self.padded_input_a: np.ndarray = None
        self.dldw: np.ndarray = None
        self.dldb: np.ndarray = None


    def forward_propagation(self, input_a: np.ndarray) -> np.ndarray:
        batch_size, input_channels, height, width = input_a.shape

        assert self.input_channels == input_channels, "Defined input channels dont match passed input channels"
        assert self.padding > -1 and self.stride > -1 and self.kernel_size > 0, "Padding, stride or kernal size is invalid"

        output_height: int = int((height + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        output_width: int = int((width + (2 * self.padding) - self.kernel_size) / self.stride) + 1

        output = np.zeros((
            batch_size,
            self.output_channels,
            output_height,
            output_width,
        ))

        self.padded_input_a = np.pad(input_a,
                            pad_width= (
                                (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)
                            ),
                            mode= "constant",
                            constant_values=0,
                            )

        for batch in range(batch_size):
            for output_channel in range(self.output_channels):
                for row in range(output_height):
                    for column in range(output_width):
                        row_start = row * self.stride
                        column_start = column * self.stride
                        neuron_a = (np.sum((
                                self.padded_input_a[
                                batch,
                                    :,
                                    row_start : row_start + self.kernel_size,
                                    column_start : column_start + self.kernel_size
                                ] * self.kernel_w[output_channel]
                            )) + self.kernel_b[output_channel]
                        )
                        output[batch, output_channel, row, column] = neuron_a

        return output

    def backpropagation(self, dldz: np.ndarray) -> np.ndarray:
        assert self.padded_input_a is not None, "dzdw is None"

        batch_size, output_channels, output_height, output_width = dldz.shape
        dzda = self.kernel_w
        dzdw = self.padded_input_a
        self.dldw = np.zeros_like(self.kernel_w)
        self.dldb = np.zeros_like(self.kernel_b)
        padded_dlda = np.zeros_like(self.padded_input_a)

        for batch in range(batch_size):
            for output_channel in range(self.output_channels):
                for row in range(output_height):
                    for column in range(output_width):
                        neuron_dldz = dldz[batch, output_channel, row, column]
                        row_start = row * self.stride
                        column_start = column * self.stride
                        self.dldw[output_channel] += (
                            neuron_dldz *
                            dzdw[
                                batch, :,
                                row_start : row_start + self.kernel_size,
                                column_start : column_start + self.kernel_size
                            ]
                        )

                        self.dldb[output_channel] += neuron_dldz

                        padded_dlda[
                            batch, :,
                            row_start : row_start + self.kernel_size,
                            column_start : column_start + self.kernel_size
                        ] += (
                            neuron_dldz
                             * dzda[output_channel]
                        )

        if self.padding > 0:
            dlda = padded_dlda[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dlda = padded_dlda

        return dlda