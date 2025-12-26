"""ALIE Attack Implementation."""

import numpy as np
from .base import AttackClient
from cifar_cnn.task import get_parameters

class ALIEClient(AttackClient):
    """Attack 6: A Little Is Enough (ALIE).
    Tấn công thống kê dựa trên phân phối chuẩn.
    Công thức: w_mal = w_mean - z * std
    Lưu ý: Trong Simulation Client, ta không biết w_mean và std toàn cục.
    Ta sử dụng local update như một ước lượng (estimator) hoặc giả lập hành vi này.
    """
    
    def __init__(self, net, trainloader, testloader, device, local_epochs,
                 learning_rate=0.001, use_mixed_precision=True, proximal_mu=0.01,
                 z=3, mode="alie"):
        super().__init__(net, trainloader, testloader, device,
                        local_epochs, learning_rate, use_mixed_precision, proximal_mu)
        self.z = z

    def fit(self, parameters, config):
        # 1. Train bình thường
        results = self.train_with_fedprox(parameters)
        trained_params = get_parameters(self.net)
        global_params = parameters
        
        malicious_params = []
        
        # ALIE Logic (Local Approximation):
        # Ta muốn tạo ra một update nằm ở "rìa" của phân phối chấp nhận được.
        # Giả sử update cục bộ là u. Ta sẽ biến đổi u.
        
        for w_g, w_t in zip(global_params, trained_params):
            if not isinstance(w_t, np.ndarray):
                w_t = np.array(w_t)
            
            # Update vector
            u = w_t - w_g
            
            # Tính std của chính update vector này (layer-wise estimation)
            # Đây là heuristic vì client không biết std của các clients khác.
            sigma = np.std(u)
            
            # Công thức ALIE: u_mal = u - z * sigma
            # Di chuyển update về phía ngược lại một khoảng z*sigma
            u_mal = u - (self.z * sigma)
            
            # W_mal = W_global + u_mal
            w_mal = w_g + u_mal
            malicious_params.append(w_mal)
        results["is_malicious"] = 1    
        return malicious_params, len(self.trainloader.dataset), results