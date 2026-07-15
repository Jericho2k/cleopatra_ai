# Writer naturalness prompt tune

This is a prompt-only calibration. It does not change routing, commercial policy,
conversation state, temperature, or model selection.

The live failure was strategically acceptable but linguistically artificial:

```text
lucky you, Tony
```

The previous prompt already prohibited that exact phrase while also placing it in
the prompt as a negative example. Language models can still copy negative examples,
and the surrounding instruction rewarded distinctive teasing and "light friction."

The tune therefore:

- removes literal stock-line examples from the stable prompt;
- makes option 1 the deliberately plain auto-send default;
- requires direct acknowledgment of the latest message;
- prefers ordinary specific wording over cleverness;
- adds specificity and "sounds typed, not authored" self-checks;
- keeps options 2 and 3 available for warmer or bolder variation;
- leaves Kimi as the ordinary-chat writer so the result can be evaluated cleanly.

After deployment, test several fresh-fan openings rather than one conversation.
The target is not that every line becomes bland. The target is that personality
comes from the creator persona and exact context instead of canned banter.
