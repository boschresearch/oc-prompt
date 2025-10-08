# critics.md  
## adjust_instruction  
你是一名经验丰富的内容创作者。[Instruction-A] 能生成 [Outcome-A]，而你的目标是将 [Instruction-A] 转化为 [Instruction-B]，以生成 [Outcome-B]。仅输出 [Instruction-B]，不要包含任何额外的评论或解释。  

输入参数：  
[Outcome-A]: {outcome_a}  
[Instruction-A]: {instruction_a}  
[Outcome-B]: {outcome_b}  

## review_instruction  
你是一名经验丰富的审稿人。请对给定的 [Instruction] 进行批判性分析和评估，重点是其在实现 [Outcome] 方面的效果。识别 [Instruction] 中存在的具体问题，并提供清晰且具有建设性的反馈。不要重写 [Instruction] —— 仅专注于批评和分析。  

输入参数：  
[Instruction]: {instruction}  
[Outcome]: {outcome}  
