import torch



def make_grid2d(x, y):
    xx = y.view(len(y), 1).repeat(1, len(x))
    yy = x.repeat(len(y), 1)
    return torch.stack((xx,yy))




class GridLayer3d:
    def __init__(self,  shape, x_min=-1,x_max=1, y_min=-1, y_max=1):
        # shape = (y, x)
        self.shape = shape
        self._x = torch.linspace(x_min, x_max, shape[0])
        self._y = torch.linspace(y_min, y_max, shape[1])

    def get_cords(self):
        # returns tensor of size (2, len(y), len(x))
        return  make_grid2d(self._y, self._x)





class Substrate3D:
    def __init__(self, layers):
        self._layers = layers
        self._calculate_layer_cords()
        self._calculate_weight_cords()

    def _get_z_cords(self):
        return torch.linspace(-1, 1, len(self._layers))

    def _calculate_layer_cords(self):
        self._layer_cords= []
        for z, l in zip(self._get_z_cords(), self._layers):
            zz  = torch.full((1, *l.shape), z)
            self._layer_cords.append(torch.cat((zz, l.get_cords()), dim=0))


    def _calculate_weight_cords(self):
        self._weight_cords = []
        for l1, l2 in zip(self._layer_cords[:-1], self._layer_cords[1:]):
            make_grid2d(l1, l2)


    def _define_weights(self):
        self._weights = []
        pass


    def set_weights(self, cppn):
        pass

    def forward(self):
        pass



if __name__ == '__main__':
    net = Substrate3D([GridLayer3d((10,6)), GridLayer3d((5,3))])
