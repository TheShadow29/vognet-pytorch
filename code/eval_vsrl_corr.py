"""
Better evaluation.
Corrected a few implementations
for sep, temporal, spatial.
"""
from eval_fn_corr import (
    GroundEvalDS4_Sep,
    GroundEvalDS4_Temporal,
    GroundEvalDS4_Spatial
)
import pickle
from fastprogress import progress_bar

from pathlib import Path
import torch
from trn_utils import (compute_avg_dict,
                       is_main_process,
                       synchronize,
                       get_world_size)


class Evaluator(torch.nn.Module):
    def __init__(self, cfg, comm, device):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.met_keys = ['avg1', 'macro_avg1']
        self.num_prop_per_frm = self.comm.num_prop_per_frm
        self.num_frms = self.cfg.ds.num_sampled_frm
        self.num_props = self.num_prop_per_frm * self.num_frms
        self.device = device
        self.after_init()

    def after_init(self):
        pass

    def get_out_results(self, out_result):
        if isinstance(out_result, torch.Tensor):
            return out_result
        else:
            return out_result['mdl_outs']

    def forward_one_batch(self, out_result, inp):
        """
        The following should be returned:
        List[Dict]
        Dict = {
            'idx(video)', 'idx(srl)', 'idx(arg)',
            'pred_boxes', 'pred_scores'
        }
        """
        out_result = out_result
        # B x num_verbs x num_srl_args x 1000
        B, num_verbs, num_srl_args, num_props = out_result.shape
        assert self.num_props == num_props
        # B x num_verbs x num_srl_args x num_frms x num_prop_per_frm
        out_result_frame = torch.sigmoid(
            out_result.view(
                B, num_verbs, num_srl_args,
                self.num_frms, self.num_prop_per_frm
            )
        )
        # B x num_verbs x num_srl_args x num_frms x num_prop_per_frm
        out_result_frame_score, out_result_frame_index = torch.max(
            out_result_frame, dim=-1)

        props = inp['pad_proposals']
        _, num_props, prop_dim = props.shape
        # B x num_verbs x num_srl_args x num_frms x num_prop_per_frm x prop_dim
        props_reshaped = props.view(
            B, 1, 1, self.num_frms, self.num_prop_per_frm, prop_dim).expand(
                B, num_verbs, num_srl_args,
                self.num_frms, self.num_prop_per_frm, prop_dim)

        out_result_boxes = torch.gather(
            props_reshaped, dim=-2,
            index=out_result_frame_index.unsqueeze(-1).unsqueeze(-1).expand(
                *out_result_frame_index.shape, 1, prop_dim))

        pred_boxes = out_result_boxes.squeeze(-2)
        out_dict_list = [
            {
                'pred_boxes': pb,
                'pred_scores': ps,
                'idx_vid': an_ind,
                'idx_sent': srl_ann,
                'idx_verb': srl_verb,
                'num_verbs': nv

            } for pb, ps, an_ind, srl_ann, srl_verb, nv in zip(
                pred_boxes.detach().cpu().tolist(),
                out_result_frame_score.detach().cpu().tolist(),
                inp['ann_idx'].detach().cpu().tolist(),
                inp['sent_idx'].detach().cpu().tolist(),
                inp['srl_verb_idxs'].detach().cpu().tolist(),
                inp['num_verbs'].detach().cpu().tolist()
            )]
        return out_dict_list

    def forward(self, model, loss_fn, dl, dl_name,
                rank=0, pred_path=None, mb=None):
        fname = Path(pred_path) / f'{dl_name}_{rank}.pkl'
        # comm = self.comm
        # cfg = self.cfg
        model.eval()
        loss_keys = loss_fn.loss_keys
        val_losses = {k: [] for k in loss_keys}
        nums = []
        results = []
        for batch in progress_bar(dl, parent=mb):
            for b in batch.keys():
                batch[b] = batch[b].to(self.device)
            b = next(iter(batch.keys()))
            nums.append(batch[b].size(0))
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model(batch)
                out_loss = loss_fn(out, batch)

            for k in out_loss:
                val_losses[k].append(out_loss[k].detach().cpu())
            results += self.forward_one_batch(out, batch)

        pickle.dump(results, open(fname, 'wb'))
        nums = torch.tensor(nums).float()
        val_loss = compute_avg_dict(val_losses, nums)

        synchronize()
        if is_main_process():
            curr_results = results
            world_size = get_world_size()
            for w in range(1, world_size):
                tmp_file = Path(pred_path) / f'{dl_name}_{w}.pkl'
                with open(tmp_file, 'rb') as f:
                    tmp_results = pickle.load(f)
                curr_results += tmp_results
                tmp_file.unlink
            with open(fname, 'wb') as f:
                pickle.dump(curr_results, f)
            out_acc = self.grnd_eval.eval_ground_acc(fname)
            val_acc = {k: torch.tensor(v).to(self.device)
                       for k, v in out_acc.items() if k in self.met_keys}
            # return val_loss, val_acc
        synchronize()
        if is_main_process():
            return val_loss, val_acc
        else:
            return {k: torch.tensor(0.).to(self.device) for k in loss_keys}, {
                k: torch.tensor(0.).to(self.device) for k in self.met_keys}


class EvaluatorDS4_Corr_SSJ1_Sep(Evaluator):
    def after_init(self):
        # self.met_keys = ['avg1', 'macro_avg1', 'avg1_cons', 'macro_avg1_cons']
        # self.grnd_eval = GroundEvalDS4(self.cfg, self.comm)

        self.met_keys = ['avg1', 'avg1_cons',
                         'avg1_vidf', 'avg1_strict']
        self.grnd_eval = GroundEvalDS4_Sep(self.cfg, self.comm)

        self.num_sampled_frm = self.num_frms
        # self.num_prop_per_frm = self.cfg.misc.num_prop_per_frm

    def get_out_results_boxes(self, out_result_dict, inp):
        """
        get the correct boxes, scores, indexes per frame
        """
        assert isinstance(out_result_dict, dict)
        # B x num_cmp
        fin_scores = out_result_dict['fin_scores']
        B, num_cmp = fin_scores.shape

        # B
        vidf_outs = torch.argmax(fin_scores, dim=-1)

        # B x num_cmp x num_srl_args x num_props
        mdl_outs = out_result_dict['mdl_outs_eval']

        B, num_cmp, num_srl_args, num_props = mdl_outs.shape

        mdl_outs_reshaped = mdl_outs.transpose(
            1, 2).contiguous().view(
                B, num_srl_args, num_cmp,
                self.num_sampled_frm, self.num_prop_per_frm
        )

        # B x num_srl_args x num_cmp x num_frms
        out_result_frame_score, out_result_frame_index = torch.max(
            mdl_outs_reshaped, dim=-1
        )

        props = inp['pad_proposals']
        _, num_cmp, num_props, prop_dim = props.shape

        props_reshaped = props.view(
            B, 1, num_cmp,
            self.num_sampled_frm, self.num_prop_per_frm, prop_dim
        ).expand(
            B, num_srl_args, num_cmp,
            self.num_sampled_frm, self.num_prop_per_frm, prop_dim
        )

        props_out = torch.gather(
            props_reshaped,
            dim=-2,
            index=out_result_frame_index.unsqueeze(-1).unsqueeze(-1).expand(
                B, num_srl_args, num_cmp,
                self.num_sampled_frm, 1, prop_dim
            )
        )

        props_out = props_out.squeeze(-2)

        # B -> B x #srl x #frms
        vidf_outs = vidf_outs.view(B, 1, 1).expand(
            B, num_srl_args, self.num_frms)

        return {
            'boxes': props_out,
            'scores': out_result_frame_score,
            'indexs': vidf_outs
        }

    def forward_one_batch(self, out_result, inp):
        """
        The following should be returned:
        List[Dict]
        Dict = {
            'idx(video)', 'idx(srl)', 'idx(arg)',
            'pred_boxes', 'pred_scores'
        }
        """
        out_results = self.get_out_results_boxes(out_result, inp)

        out_result_boxes = out_results['boxes']
        out_result_frame_score = out_results['scores']
        out_result_frame_index = out_results['indexs']

        # B x num_srl_args x num_cmp x num_frms x num_props
        pred_boxes = out_result_boxes
        # B x num_srl_args x num_frms
        pred_cmp = out_result_frame_index
        # B x num_srl_args x num_cmp x num_frms
        pred_score = out_result_frame_score
        targ_cmp = inp['target_cmp'].detach().cpu().tolist()
        perm_list = inp['permute'].detach().cpu().tolist()
        perm_inv_list = inp['permute_inv'].detach().cpu().tolist()
        # targ_cmp = inp['target_cmp'][0].item()
        # perm = inp['permute'][0].detach().cpu().tolist()
        # perm_inv = inp['permute_inv'][0].detach().cpu().tolist()

        out_dict_list = [
            {
                'pred_boxes': pb,
                'pred_scores': ps,
                'pred_cmp': pc,
                'idx_vid': an_ind,
                'idx_verbs': srl_idxs,
                'idx_sent': srl_ann,
                'cmp_msk': cmp_msk,
                'targ_cmp': tcmp,
                'perm': perm,
                'perm_inv': perm_inv,

            } for pb, ps, pc, an_ind, srl_idxs, srl_ann, cmp_msk,
            tcmp, perm, perm_inv in zip(
                pred_boxes.detach().cpu().tolist(),
                pred_score.detach().cpu().tolist(),
                pred_cmp.detach().cpu().tolist(),
                inp['ann_idx'].detach().cpu().tolist(),
                inp['new_srl_idxs'].detach().cpu().tolist(),
                inp['sent_idx'].detach().cpu().tolist(),
                inp['num_cmp_msk'].detach().cpu().tolist(),
                targ_cmp,
                perm_list,
                perm_inv_list
            )]
        return out_dict_list


class EvaluatorDS4_Corr_SSJ1_Temporal(EvaluatorDS4_Corr_SSJ1_Sep):
    def after_init(self):
        # self.met_keys = ['avg1', 'macro_avg1', 'avg1_cons', 'macro_avg1_cons']
        # self.grnd_eval = GroundEvalDS4(self.cfg, self.comm)

        self.met_keys = ['avg1', 'avg1_cons',
                         'avg1_vidf', 'avg1_strict']
        self.grnd_eval = GroundEvalDS4_Temporal(self.cfg, self.comm)

        # self.num_sampled_frm = self.cfg.misc.num_sampled_frm
        self.num_sampled_frm = self.num_frms
        # self.num_prop_per_frm = self.cfg.misc.num_prop_per_frm

    def get_out_results_boxes(self, out_result_dict, inp):
        """
        get the correct boxes, scores, indexes per frame
        """
        assert isinstance(out_result_dict, dict)

        out_result = out_result_dict['mdl_outs_eval']
        num_cmp = inp['new_srl_idxs'].size(1)

        # B x num_verbs x num_srl_args x 4000
        B, num_verbs, num_srl_args, num_props = out_result.shape

        assert num_verbs == 1
        # B x num_srl_args x num_props
        # mdl_outs = out_result.squeeze(1)
        mdl_outs_reshaped = out_result.view(
            B, num_srl_args, num_cmp,
            self.num_sampled_frm, self.num_prop_per_frm
        )

        # B x num_srl_args x num_cmp x num_frms
        out_result_frame_score, out_result_frame_index = torch.max(
            mdl_outs_reshaped, dim=-1
        )

        props = inp['pad_proposals']

        _, num_props, prop_dim = props.shape
        assert (num_cmp * self.num_sampled_frm *
                self.num_prop_per_frm == num_props)
        props_reshaped = props.view(
            B, 1, num_cmp,
            self.num_sampled_frm, self.num_prop_per_frm, prop_dim
        ).expand(
            B, num_srl_args, num_cmp,
            self.num_sampled_frm, self.num_prop_per_frm, prop_dim
        )

        props_out = torch.gather(
            props_reshaped,
            dim=-2,
            index=out_result_frame_index.unsqueeze(-1).unsqueeze(-1).expand(
                B, num_srl_args, num_cmp,
                self.num_sampled_frm, 1, prop_dim
            )
        )

        props_out = props_out.squeeze(-2)
        # Not used in temporal. Make it all zeros
        vidf_outs = torch.zeros(1, 1, 1).expand(
            B, num_srl_args, self.num_frms
        )
        return {
            'boxes': props_out,
            'scores': out_result_frame_score,
            'indexs': vidf_outs
        }


class EvaluatorDS4_Corr_SSJ1_Spatial(EvaluatorDS4_Corr_SSJ1_Sep):
    def after_init(self):
        self.met_keys = ['avg1', 'avg1_cons', 'avg1_vidf', 'avg1_strict']
        self.grnd_eval = GroundEvalDS4_Spatial(self.cfg, self.comm)

        self.num_sampled_frm = self.num_frms
        # self.num_sampled_frm = self.cfg.misc.num_sampled_frm
        # self.num_prop_per_frm = self.cfg.misc.num_prop_per_frm

    def get_out_results_boxes(self, out_result_dict, inp):
        """
        get the correct boxes, scores, indexes per frame
        """
        assert isinstance(out_result_dict, dict)

        out_result = out_result_dict['mdl_outs_eval']
        num_cmp = inp['new_srl_idxs'].size(1)

        # B x num_verbs x num_srl_args x 4000
        B, num_verbs, num_srl_args, num_props = out_result.shape

        assert num_verbs == 1
        # B x num_srl_args x num_props
        mdl_outs_reshaped = out_result.view(
            B, num_srl_args,
            self.num_sampled_frm, num_cmp, self.num_prop_per_frm
        )

        # B x num_srl_args x num_frm x num_cmp
        out_result_frame_score, out_result_frame_index = torch.max(
            mdl_outs_reshaped, dim=-1
        )

        props = inp['pad_proposals']

        _, num_props, prop_dim = props.shape
        assert (num_cmp * self.num_sampled_frm *
                self.num_prop_per_frm == num_props)
        props_reshaped = props.view(
            B, 1, self.num_sampled_frm,
            num_cmp, self.num_prop_per_frm, prop_dim
        ).expand(
            B, num_srl_args, self.num_sampled_frm,
            num_cmp, self.num_prop_per_frm, prop_dim
        )

        props_out = torch.gather(
            props_reshaped,
            dim=-2,
            index=out_result_frame_index.unsqueeze(-1).unsqueeze(-1).expand(
                B, num_srl_args, self.num_sampled_frm, num_cmp, 1, prop_dim
            )
        )

        # B x num_srl x num_frms x num_cmp
        props_out = props_out.squeeze(-2)
        # For consistency across sep, temporal, spatial
        props_out = props_out.transpose(2, 3).contiguous()

        # Used in spatial.
        # Divide by 100
        # vidf_outs = torch.div(
        # out_result_frame_index.squeeze(-1),
        # self.num_prop_per_frm
        # ).long()

        # B x num_srl_args x num_frm
        vidf_outs = out_result_frame_score.argmax(dim=-1)

        # B x num_srl_args x num_frm x num_cmp
        score_out = out_result_frame_score.transpose(2, 3).contiguous()

        return {
            'boxes': props_out,
            'scores': score_out,
            'indexs': vidf_outs
        }
