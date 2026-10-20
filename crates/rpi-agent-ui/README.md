# rpi-agent-ui

Reusable model-aware selectors for products embedding the `rpi` agent runtime.
The crate builds provider, model, and thinking choices from upstream metadata
and renders them with `rpi-tui` components. Hosts attach their own selection
callbacks to the returned lists.

```toml
[dependencies]
rpi-agent-ui = "0.6.2"
```

The crate does not own provider configuration, credentials, or execution
selection policy. Those remain with the upstream provider runtime and the
embedding product's application semantics.

## Controls

- `model_selector` builds a searchable list from `rpi-ai::Model` values and
  keeps provider-qualified ids distinct.
- `provider_selector` builds a searchable list from safe provider display
  options supplied by the provider runtime.
- `thinking_selector` builds a list from the thinking levels supported by an
  upstream model.

`rpi-cli` uses these controls and attaches its product actions to their
`rpi-tui` lists. Embedded products can attach their own callbacks without
depending on `rpi-cli` or maintaining a parallel selector framework.

## License

MIT. See the repository `LICENSE` and `NOTICE` files.
