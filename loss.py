import torch.nn.functional as F
import torch.nn as nn
import torch
class Loss(nn.Module):
    def __init__(self, tau=0.07,out_dim=1024,center_momentum=0.995):
        super().__init__()
        self.tau=tau
        self.center_momentum=center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output_ori, teacher_output_ori):
        """
        NCELoss of the teacher and student networks.
        student_out: torch.tensor, shape (batch*n_crops,out_dim)
        teacher_output: torch.tensor, shape (batch*n_crops,out_dim)
        """
        teacher_output=torch.cat(teacher_output_ori,dim=1)
        student_output=torch.cat(student_output_ori,dim=1)
        n_examples,_=student_output.size()
        teacher=F.normalize(teacher_output,dim=-1)
        student=F.normalize(student_output,dim=-1)
        scores=torch.mm(teacher,student.t()).div_(self.tau)
        target=torch.arange(n_examples,dtype=torch.long).to(scores.device)
        loss=F.cross_entropy(scores,target)
        self.update_center(teacher_output_ori)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    
def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients. Used to avoid gradient exponential
    Parameters
    ----------
    model : nn.Module
        Module.
    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)