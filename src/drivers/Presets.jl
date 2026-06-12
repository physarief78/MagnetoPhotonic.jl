function gdfeco_pump_probe(; overrides...)
    defaults = (;
        grid=GridConfig(),
        source=SourceConfig(),
        probe=ProbeConfig(),
        device=DeviceConfig(),
        pml=PMLConfig(),
        model=ModelConfig(),
        output=OutputConfig(),
        steps=10,
        backend=BackendConfig(),
        precision=EM_FIELD_STORAGE_TYPE,
    )
    merged = merge(defaults, (; overrides...))
    return SimConfig(; merged...)
end
