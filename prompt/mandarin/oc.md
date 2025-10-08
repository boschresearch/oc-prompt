# oc.md  
## expand_instruction  
你是一位经验丰富的内容创作者。[Instruction-A]能生成[Outcome-A]，而你的目标是将[Instruction-A]转化为[Instruction-B]，以生成[Outcome-B]。无需考虑[Instruction-B]的格式、结构、风格、语气或长度。但你的输出必须是一个全面扩展版的[Instruction-B]，其长度至少是原[Instruction-A]的1000倍。你需要整合所有相关知识、信息、最佳实践、重要注意事项，以及需避免的警告或陷阱——本质上，你需要囊括关于该主题、指令及结果的所有内容。仅输出[Instruction-B]本身，不要包含任何额外的评论或解释。  

输入参数：  
- [Instruction-A]: {instruction_a}  
- [Outcome-A]: {outcome_a}  
- [Outcome-B]: {outcome_b}  

## compact_instruction  
你是一位经验丰富的内容编辑。将[Instruction-B]（包含实现[Outcome-B]的详细步骤）精简为仅包含实现[Outcome-B]所必需的关键步骤和细节的版本。仅输出精简后的版本，不要包含任何额外的评论或解释。  

输入参数：  
- [Outcome-B]: {outcome_b}  
- [Instruction-B]: {instruction_b}  

## transfer_instruction  
你是一位经验丰富的内容编辑。[Instruction-B]是生成[Outcome-B]的指令。重写[Instruction-B]，使其在长度、风格、语气和格式上与[Instruction-A]保持一致，同时仍能生成[Outcome-B]。仅输出重写后的版本，不要包含任何额外的评论或解释。  

输入参数：  
- [Instruction-A]: {instruction_a}  
- [Instruction-B]: {instruction_b}  
- [Outcome-B]: {outcome_b}  
