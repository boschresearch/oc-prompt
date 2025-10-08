# pair_template_generate.md
## generate
You are given a structured instruction for a repair or alteration task. Each instance comprises a **Title**, **Toolbox**, **Steps**, and other metadata. Your objective is to produce a plausible **variant** of this task by altering **one major component** (e.g., replacing **one material**, **object**, or **subject** being fixed) with a similar but distinct alternative.

To achieve this:
1. **Identify** the original task’s key component that drives the repair (e.g., “ribbon button”, “watch crystal”, “jean patch”).
2. **Ensure compatibility and distinctness of the alternative component**:
   - The alternative component **exists within the same device model** or is **commonly used** in similar models of the device.
   - The alternative component is part of the **same repair domain** as the original (e.g., screen types, buttons, straps).
   - **Do not introduce new component types** that are not present in the device’s specifications.
   - **Do not select different variants or models of the same component type** (e.g., avoid replacing a specific camera model with another camera model).
   - **Do not select components that serve the same or very similar functions** as the original component (e.g., avoid replacing thermal paste with thermal pad).
   - **Infer the device’s available components** based on the original structured instruction to avoid selecting non-existent parts.
3. **Select a distinct alternative component** that:
   - Belongs to the same repair domain but is **substantially different** from the original component (avoiding near-synonyms or identical functions).
   - Is a **different component type** within the same domain, ensuring it performs a distinct function (e.g., replacing a thermal paste with a cooling fan instead of another thermal material).
   - Would entail **non-trivial adjustments** to the existing steps (e.g., different attachment method, adhesive, cutting technique).
   - **Actually exists** in the marketplace and is **commonly used** in real-world repairs for the specified device.
   - **Is not overly similar** to the original component to ensure meaningful variation.
4. **Compose** a new **Title** that reflects the variant task and names the new component unambiguously.

**Key requirements**  
- **Always** include the *complete* `base_steps` list.  
- **Always** verify that the alternative component is a valid and distinct part of the specified device to avoid introducing non-existent or nearly identical elements.

**Input Parameter:**
- structured instruction: {task}