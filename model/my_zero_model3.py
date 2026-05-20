import torch
import torch.nn as nn
import torch.optim as optim
# user defined
from src.optimizer import SAM
from .linear_module import EmbeddingNet
from .GaussianMapper import GaussianMapper, GaussianMapper2
from myloss.loss import compute_kl_divergence, compute_kl_divergence2
import torch.nn.functional as F


torch.set_printoptions(threshold=10_000) 
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class My_zero_shot_model3(nn.Module):
    def __init__(self, args,):
        super(My_zero_shot_model3, self).__init__()

        #loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()

        #optimizer
        self.lr_scheduler = args.lr_scheduler

        #other
        self.modality = args.modality  # both

        #Architecture

        mapper_hidden_dim=args.mapper_hidden_size
        enc_hidden_size=args.enc_hidden_size
        dropout=args.dropout
        output_model_dim=args.output_model_dim
        self.video_audio_proj = EmbeddingNet(input_size=1024, output_size=512, dropout=dropout, use_bn=True)
        self.video_audio_mapper = GaussianMapper2(input_dim=512, hidden_dim=mapper_hidden_dim, output_dim=output_model_dim,
                                                 dropout=dropout)

        self.text_video_audio_proj = EmbeddingNet(input_size=1024, output_size=512, dropout=dropout, use_bn=True)
        self.text_video_audio_enc = EmbeddingNet(input_size=512, hidden_size=enc_hidden_size, output_size=output_model_dim,
                                                 dropout=dropout, use_bn=True)

        # Optimizer
        self.lr = args.lr
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(),lr=self.lr, weight_decay=args.weight_decay)

        #lr_scheduler
        if self.lr_scheduler=="reduce":
            self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                                patience=3, verbose=True)
        if self.lr_scheduler == 'cosine':
            self.scheduler_learning_rate=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=args.epochs,eta_min=1e-6)

        print('Done')

    def optimize_scheduler(self, value):
        if self.lr_scheduler=='reduce':
            self.scheduler_learning_rate.step(value)
        elif self.lr_scheduler == 'cosine':
            self.scheduler_learning_rate.step()

    def optimize_params(self,
                        audio_embeddings,
                        video_embeddings,
                        targets,
                        text_embeddings,
                        text_embeddings_all_current,
                        args,
                        optimize=False,
                        **kwargs):

        outputs = self.forward(audio_embeddings, video_embeddings, text_embeddings,**kwargs)

        loss_total, loss_details = self.compute_loss(outputs, text_embeddings_all_current, targets, args,optimize)

        if optimize == True:
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

        return loss_total, loss_details, self.optimizer.param_groups[0]['lr']



    def forward(self, audio, video, text_video_audio, **kwargs):
        
        video = video.type(torch.float32)
        text_video_audio=text_video_audio.type(torch.float32)

        box_jpg=kwargs.get('box_jpg_embeddings')
        box_wav=kwargs.get('box_wav_embeddings')
        if box_jpg is not None and box_wav is not None:
            box_jpg, box_wav= box_jpg.type(torch.float32), box_wav.type(torch.float32)
        video_audio_box=torch.cat((box_jpg,box_wav),dim=-1) if box_jpg is not None and box_wav is not  None else None

        video_audio=torch.cat((video,audio),dim=-1)

        #video_audio forward
        video_audio = self.video_audio_proj(video_audio)
        video_audio_mu, video_audio_sigma = self.video_audio_mapper(video_audio)

        #video_audio_box_forward
        if video_audio_box is not None:
            video_audio_box = self.video_audio_proj(video_audio_box)
            video_audio_box_mu, video_audio_box_sigma = self.video_audio_mapper(video_audio_box)

        #text forward
        text_video_audio = self.text_video_audio_proj(text_video_audio)
        text_video_audio = self.text_video_audio_enc(text_video_audio)


        outputs = {
            "video_audio_mu":video_audio_mu,
            "video_audio_sigma":video_audio_sigma,

            "video_audio_box_mu":video_audio_box_mu if video_audio_box is not None else None,
            "video_audio_box_sigma":video_audio_box_sigma if video_audio_box is not None else None,

            "text_video_audio":text_video_audio,
        }
        return outputs

    def compute_loss(self, outputs, text_embeddings_all_train, target,args,optimize):

        video_audio_mu = outputs['video_audio_mu']
        video_audio_sigma = outputs['video_audio_sigma']


        video_audio_box_mu = outputs.get('video_audio_box_mu')
        video_audio_box_sigma = outputs.get('video_audio_box_sigma')


        text_embeddings_all_train = self.text_video_audio_proj(text_embeddings_all_train)
        text_embeddings_all_train = self.text_video_audio_enc(text_embeddings_all_train)

        #video_audio
        scores_video_audio = torch.matmul(video_audio_mu, text_embeddings_all_train.t())
        l_ce_video_audio = self.criterion_cls(scores_video_audio, target)

        #box
        if  video_audio_box_mu is not None:

            text_video_audio=outputs.get("text_video_audio")
            l_reg_box=self.MSE_loss(text_video_audio,video_audio_box_mu)

            l_kl_va, kl_stats = compute_kl_divergence2(
                mu_specific=video_audio_mu, sigma_specific=video_audio_sigma,
                mu_general=video_audio_box_mu, sigma_general=video_audio_box_sigma,
                margin_ratio=args.margin_ratio
            )

            l_kl_total=l_kl_va
        #ce total
        l_ce_total=l_ce_video_audio+ l_reg_box if video_audio_box_mu is not None else l_ce_video_audio

        loss_total = l_ce_total+l_kl_total  if video_audio_box_mu is not None else l_ce_total



        loss_details = {
            "total_loss": loss_total.detach().cpu(),
            "loss_ce": l_ce_video_audio.detach().cpu(),
            "loss_reg":l_reg_box if video_audio_box_mu is not None else torch.tensor(0.0) ,
            "loss_kl": l_kl_total.detach().cpu() if video_audio_box_mu is not None else torch.tensor(0.0, device=video_audio_mu.device),

            "var_video": kl_stats["var_video"].cpu() if video_audio_box_mu is not None else torch.tensor(0.0),
            "var_ratio": kl_stats["var_ratio"].cpu() if video_audio_box_mu is not None else torch.tensor(0.0),

            "mu_video_norm": kl_stats["mu_video_norm"].cpu() if video_audio_box_mu is not None else torch.tensor(0.0),
            "mu_box_norm": kl_stats["mu_box_norm"].cpu() if video_audio_box_mu is not None else torch.tensor(0.0),
            "mu_distance": kl_stats["mu_distance"].cpu() if video_audio_box_mu is not None else torch.tensor(0.0),
        }

        return loss_total, loss_details


    def get_embeddings(self, audio, video,text_embeddings_all):
        video = video.type(torch.float32)
        text_embeddings_all=text_embeddings_all.type(torch.float32)

        video_audio= torch.cat((video,audio), dim=1)

        #video_audio
        video_audio = self.video_audio_proj(video_audio)
        video_audio_mu, video_audio_sigma = self.video_audio_mapper(video_audio) 
        #text
        text_embeddings_all = self.text_video_audio_proj(text_embeddings_all)
        text_embeddings_all = self.text_video_audio_enc(text_embeddings_all)

        text_embeddings_all_outputs={
            "text_video_audio":text_embeddings_all,
        }

        return video_audio_mu,text_embeddings_all_outputs
