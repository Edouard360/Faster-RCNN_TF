from generate_data.write_xml import symbol_to_text

SYMBOLS_CLASSES = [r'$\longrightarrow$', r'$\sigma$', r'$\alpha$', r'$\gamma$',
                   r'$\int$']

TEXT_CLASSES = ('__background__',) + tuple([symbol_to_text(s) for s in SYMBOLS_CLASSES])