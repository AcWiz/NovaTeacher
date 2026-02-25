from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmrotate.utils import get_root_logger



'''

这段代码定义了一个名为 MeanTeacher 的 Hook 类，主要用于实现 Mean Teacher 模型中的 EMA (Exponential Moving Average) 参数更新。该 Hook 在训练过程中的每个迭代后，根据设定的条件进行 EMA 更新。

关键功能和属性：

momentum: EMA 更新的动量参数，用于平滑更新，默认为 0.9996。

interval: EMA 更新的间隔迭代数，每隔 interval 迭代进行一次 EMA 更新。

warm_up: EMA 更新的热身迭代数，表示在前 warm_up 迭代内不进行 EMA 更新。

start_steps: EMA 更新的起始迭代数，从这一步开始进行 EMA 更新。

skip_buffer: 是否跳过模型中的缓冲区（buffer），默认为 True。

主要方法：

before_run: 在训练开始前进行初始化，确保模型中包含了必需的属性，如教师模型和学生模型。

after_train_iter: 在每个训练迭代之后执行 EMA 更新。首先判断是否满足 EMA 更新的条件，然后调用 momentum_update 方法进行参数更新。

momentum_update: 根据指定的动量参数，进行 EMA 更新。更新时可以选择是否跳过模型中的缓冲区，即是否更新缓冲区的参数。
'''




@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.9996,
        interval=1,
        # static_interval=3200,
        warm_up=100,
        start_steps=10000,
        skip_buffer=True
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        # momentum warm up is disabled
        self.warm_up = warm_up
        self.interval = interval
        self.start_steps = start_steps
        self.skip_buffer = skip_buffer
        
        self.static_interval = 6400

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        assert hasattr(model, "static_teacher")

    def after_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        curr_step = model.iter_count
        if curr_step % self.interval != 0 or curr_step < self.start_steps:
            return
        if curr_step == self.start_steps:
            logger = get_root_logger()
            logger.info(f"Start EMA Update at step {curr_step}")
            self.burnin_momentum_update(model, 0)
            # self.momentum_static_update(model, 0)
            
        else:
            self.momentum_update(model, self.momentum)

        if curr_step % self.static_interval == 0  & curr_step > self.static_interval:
            # 更新慢
            self.momentum_static_update(model,self.momentum)



#     def after_train_iter(self, runner):
#             model = runner.model
#             if is_module_wrapper(model):
#                 model = model.module
#             curr_step = model.iter_count
#             if curr_step % self.interval != 0 or curr_step < self.start_steps:
#                 return
#             if curr_step == self.start_steps:
#                 logger = get_root_logger()
#                 logger.info(f"Start EMA Update at step {curr_step}")
#                 self.momentum_update(model, 0)
#             else:
#                 self.momentum_update(model, self.momentum)

# #   单教师
#     def momentum_update(self, model, momentum):
#         if self.skip_buffer:
#             for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
#                 model.student.named_parameters(), model.teacher.named_parameters()
#             ):
#                 tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
#         else:
#             for (src_parm,
#                  dst_parm) in zip(model.student.state_dict().values(),
#                                   model.teacher.state_dict().values()):
#                 # exclude num_tracking
#                 if dst_parm.dtype.is_floating_point:
#                     dst_parm.data.mul_(momentum).add_(
#                         src_parm.data, alpha=1 - momentum)


#  静态双教师


    def burnin_momentum_update(self, model, momentum):
        if self.skip_buffer:
            for (src_name, src_parm), (tgt_name, tgt_parm), (stgt_name, stgt_parm) in zip(
                model.student.named_parameters(), model.teacher.named_parameters(),model.static_teacher.named_parameters()
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
        else:
            for (src_parm,
                 dst_parm) in zip(model.student.state_dict().values(),
                                  model.teacher.state_dict().values(),
                                  model.static_teacher.state_dict().values()
                                  ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(momentum).add_(
                        src_parm.data, alpha=1 - momentum)
    
    
    
    


# 动态双教师写法
    
    def momentum_update(self, model, momentum):
        if self.skip_buffer:
            for (src_name, src_parm), (tgt_name, tgt_parm), (stgt_name, stgt_parm)in zip(
                model.student.named_parameters(), model.teacher.named_parameters(),model.static_teacher.named_parameters()
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
        else:
            for (src_parm,
                 dst_parm) in zip(model.student.state_dict().values(),
                                  model.teacher.state_dict().values(),
                                #   model.static_teacher.state_dict().values()
                                  ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(momentum).add_(
                        src_parm.data, alpha=1 - momentum)
    
    
    
    def momentum_static_update(self, model, momentum):
        

        if self.skip_buffer:
            for (src_name, src_parm), (tgt_name, tgt_parm), (stgt_name, stgt_parm) in zip(
                model.student.named_parameters(), model.teacher.named_parameters(),model.static_teacher.named_parameters()
            ):
                # tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                temp_parm = src_parm.data.clone()  # 复制学生模型的参数
                src_parm.data.copy_(stgt_parm.data)  # 将静态教师模型的参数赋值给学生模型
                stgt_parm.data.copy_(temp_parm) 
        else:
            for (src_parm,
                 dst_parm) in zip(model.student.state_dict().values(),
                                #   model.teacher.state_dict().values(),
                                  model.static_teacher.state_dict().values()):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    # dst_parm.data.mul_(momentum).add_(
                    #     src_parm.data, alpha=1 - momentum)
                    temp_param = src_parm.data.clone()  
                    src_parm.data.copy_(dst_parm.data)  
                    dst_parm.data.copy_(temp_param)  




# 消融，动态双教师的另外写法。
# # 动态双教师写法
    
    # def momentum_update(self, model, momentum):
    #     if self.skip_buffer:
    #         for (src_name, src_parm), (tgt_name, tgt_parm), (stgt_name, stgt_parm)in zip(
    #             model.student.named_parameters(), model.teacher.named_parameters(),model.static_teacher.named_parameters()
    #         ):
    #             stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
    #             # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
    #     else:
    #         for (src_parm,
    #              dst_parm) in zip(model.student.state_dict().values(),
    #                             #   model.teacher.state_dict().values(),
    #                               model.static_teacher.state_dict().values()
    #                               ):
    #             # exclude num_tracking
    #             if dst_parm.dtype.is_floating_point:
    #                 dst_parm.data.mul_(momentum).add_(
    #                     src_parm.data, alpha=1 - momentum)
    
    
    
    # def momentum_static_update(self, model, momentum):
        

    #     if self.skip_buffer:
    #         for (src_name, src_parm), (tgt_name, tgt_parm), (stgt_name, stgt_parm) in zip(
    #             model.student.named_parameters(), model.teacher.named_parameters(),model.static_teacher.named_parameters()
    #         ):
    #             # tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
    #             # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
    #             temp_parm = src_parm.data.clone()  # 复制学生模型的参数
    #             src_parm.data.copy_(tgt_parm.data)  # 将静态教师模型的参数赋值给学生模型
    #             tgt_parm.data.copy_(temp_parm) 
                
    #             tgt_parm.data.copy_(src_parm.data) 
    #     else:
    #         for (src_parm,
    #              dst_parm) in zip(model.student.state_dict().values(),
    #                               model.teacher.state_dict().values(),
    #                             #   model.static_teacher.state_dict().values()
    #                             ):
    #             # exclude num_tracking
    #             if dst_parm.dtype.is_floating_point:
    #                 # dst_parm.data.mul_(momentum).add_(
    #                 #     src_parm.data, alpha=1 - momentum)
    #                 temp_param = src_parm.data.clone()  
    #                 src_parm.data.copy_(dst_parm.data)  
    #                 dst_parm.data.copy_(temp_param)  
