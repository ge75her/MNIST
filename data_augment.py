import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class DataAugmentation:
    def __init__(self,global_crops_scale=(0.4,1),local_crops_scale=(0.05,0.4),n_local_crops=2,output_size=112):

        self.n_local_crops=n_local_crops
        RandomGaussianBlur=lambda p: transforms.RandomApply([transforms.GaussianBlur(kernel_size=1,sigma=(0.1,2))],p=p)
        flip_and_rotation=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=(10)),])
        normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),])


        self.global_1=transforms.Compose([
            transforms.RandomResizedCrop(output_size,scale=global_crops_scale,interpolation=InterpolationMode.BICUBIC),
            flip_and_rotation,
            RandomGaussianBlur(1.0),
            normalize
        ])
        self.global_2=transforms.Compose([
            transforms.RandomResizedCrop(output_size,scale=global_crops_scale,interpolation=InterpolationMode.BICUBIC),
            flip_and_rotation,
            RandomGaussianBlur(0.1),
            transforms.RandomSolarize(170,p=0.2),
            normalize
        ])
        self.local=transforms.Compose([
            transforms.RandomResizedCrop(output_size,scale=local_crops_scale,interpolation=InterpolationMode.BICUBIC),
            flip_and_rotation,
            RandomGaussianBlur(0.5),
            normalize
        ])

    
    def __call__(self,image):
        '''
        all_crops:list of torch.Tensor
        represent different version of input img
        include total 4 crops: 2 global crops & 2 local crops
        '''
        all_crops=[]
        all_crops.append(self.global_1(image))
        all_crops.append(self.global_2(image))
        all_crops.extend([self.local(image) for _ in range(self.n_local_crops)])
        return all_crops