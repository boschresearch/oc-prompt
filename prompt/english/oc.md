# oc.md
## expand_instruction
You are an experienced content creator. The [Instruction-A] produces the [Outcome-A], and your goal is to transform the [Instruction-A] into the [Instruction-B] to produce the [Outcome-B]. Do not worry about the format, structure, style, tone, or length of the [Instruction-B]. However, your output must be a comprehensive, expanded version of the [Instruction-B] that is at least 1000 times longer than the original [Instruction-A]. You must incorporate all relevant knowledge, information, best practices, important considerations, as well as warnings or pitfalls to avoid — essentially, everything you know about the subject, Instruction, and outcome. Output only the sole [Instruction-B] without any additional commentary or explanations.

Input Parameters:
- [Instruction-A]: {instruction_a}  
- [Outcome-A]: {outcome_a}   
- [Outcome-B]: {outcome_b}


## compact_instruction
You are an experienced content editor. Refine [Instruction-B], which includes detailed steps to achieve [Outcome-B], into the version containing only the essential steps and details necessary for [Outcome-B]. Provide the refined version as the sole output without any additional commentary or explanations.

Input Parameters:
- [Outcome-B]: {outcome_b}
- [Instruction-B]: {instruction_b}


## transfer_instruction
You are an experienced content editor. [Instruction-B] is the instruction that produces [Outcome-B]. Rewrite [Instruction-B] to match the length, style, tone, and format of [Instruction-A]. The rewritten [Instruction-B] must still produce [Outcome-B]. Provide the rewritten version as the sole output without any additional commentary or explanations.

Input parameters: 
- [Instruction-A]: {instruction_a}
- [Instruction-B]: {instruction_b} 
- [Outcome-B]: {outcome_b} 
