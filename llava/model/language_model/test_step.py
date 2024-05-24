class Test:
    def __init__(self):
        self.step = 0
        self.strideList = []
        self.pivotList = []
        self.groupingLayerList = []
    
    def set_lists(self, strideList, pivotList, groupingLayerList):
        self.strideList = strideList
        self.pivotList = pivotList
        self.groupingLayerList = groupingLayerList
        self.stride = self.strideList.pop(0)
        self.pivot = self.pivotList.pop(0)
        self.groupingLayer = self.groupingLayerList.pop(0)
        print(f"Stride reduction, present stride is {self.stride}, present grouping layer is {self.groupingLayer}, present pivot is {self.pivot}")
    
    def step_stride_and_layer(self):
        if self.step % self.pivot == 0 and self.step != 0 and self.stride > 1:
            self.stride = self.strideList.pop(0)
            self.groupingLayer = self.groupingLayerList.pop(0)
            self.pivot = self.pivotList.pop(0) if len(self.pivotList) > 0 else 10000
            print(f"Stride reduction, present stride is {self.stride}, present grouping layer is {self.groupingLayer}, present pivot is {self.pivot}")
        self.step += 1
        print(self.step)


str2list = lambda x: list(map(int, x.split(",")))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strides", type=str2list, default="8,8,1")
    parser.add_argument("--layers", type=str2list, default="2,16,33")
    parser.add_argument("--pivots", type=str2list, default="1300,2600")
    args = parser.parse_args()
    strides = args.strides
    layers = args.layers
    pivots = args.pivots
    test = Test()
    test.set_lists(strides, pivots, layers)
    for i in range(0, 5198):
        test.step_stride_and_layer()
        