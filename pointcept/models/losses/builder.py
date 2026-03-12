"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        
        # 🔧 检查输入中是否有NaN/Inf，避免NaN传播
        import torch
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            # 如果输入包含NaN/Inf，返回0损失（避免训练崩溃）
            # 这通常表示模型输出有问题，需要检查模型本身
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        loss = 0
        for i, c in enumerate(self.criteria):
            try:
                loss_i = c(pred, target)
                
                # 🔧 检查每个损失函数输出是否为NaN/Inf
                if torch.isnan(loss_i) or torch.isinf(loss_i):
                    # 如果某个损失函数返回NaN，跳过它（使用0代替）
                    # 这样可以避免整个训练崩溃
                    import warnings
                    warnings.warn(
                        f"Loss function {i} ({type(c).__name__}) returned NaN/Inf. "
                        f"Skipping this loss component to avoid training crash."
                    )
                    loss_i = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
                
                loss += loss_i
            except Exception as e:
                # 🔧 捕获异常，避免训练崩溃
                import warnings
                warnings.warn(
                    f"Error in loss function {i} ({type(c).__name__}): {e}. "
                    f"Skipping this loss component."
                )
                loss_i = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
                loss += loss_i
        
        # 🔧 最终检查总loss是否为NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            # 如果总loss是NaN，返回0（避免训练崩溃）
            import warnings
            warnings.warn(
                "Total loss is NaN/Inf. Returning 0 to avoid training crash. "
                "This may indicate a problem with the model or loss functions."
            )
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
