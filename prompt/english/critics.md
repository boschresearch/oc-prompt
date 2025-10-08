# critics.md
## adjust_instruction
You are an experienced content creator. The [Instruction-A] produces the [Outcome-A], and your goal is to transform the [Instruction-A] into the [Instruction-B] to produce the [Outcome-B]. Output only [Instruction-B] without any additional commentary or explanations.

Input Parameters:
[Outcome-A]: {outcome_a}  
[Instruction-A]: {instruction_a}  
[Outcome-B]: {outcome_b}


## review_instruction
You are an experienced reviewer. Critically analyze and evaluate the provided [Instruction] in relation to achieving the [Outcome]. Identify specific issues in the [Instruction] and provide clear, constructive feedback. Do not rewrite the [Instruction]—focus solely on critique and analysis.

Input Parameters:
[Instruction]: {instruction}  
[Outcome]: {outcome}


## revise_instruction_on_feedback
You are an experienced content editor. Rewrite [Instruction-B] based on the [Feedback] to match the length, style, tone, and format of [Instruction-A]. The rewritten [Instruction-B] must still produce [Outcome-B]. Provide only the rewritten content as the output without any additional commentary or explanation.

Input Parameters:
[Outcome-B]: {outcome_b}  
[Instruction-A]: {instruction_a}
[Instruction-B]: {instruction_b}
[Feedback]: {feedback}