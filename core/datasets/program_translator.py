"""
Translate programs of datasets to programs of reasoning models.
"""


def get_gqa_block_op(block):
    if block['operation'] == 'select':
        operation = 'select'
        argument = ''
        concept = block['argument'].split('(')[0].strip()

    elif block['operation'].startswith('filter'):
        operation = 'filter'
        if len(block['operation'].split())>1:
            argument = block['operation'].split()[1]
        else:
            argument = ''
        concept = block['argument']

    elif block['operation'] == 'relate':
        args = block['argument'].split(' (')[0]
        concept, argument, so_ = args.split(',')    # here the argument is the relation name
        if so_ == 's':
            operation = 'relate_s'
        elif so_ == 'o':
            operation = 'relate_o'
        elif so_ == '_':
            operation = 'relate_ae'
            #assert(argument.startswith('same'))
            if args.startswith('same'):
                argument = argument.split()[1]
            else:
                argument = 'color'

    elif block['operation'].startswith('choose'):
        if len(block['operation'].split())>1:
            argument = block['operation'].split()[1]    # argument can include rel
        else:
            argument = ''
        concept = block['argument']
        if len(block['dependencies']) > 1:
            operation = 'choose_in2'
        else:
            operation = 'choose_in1'
            if argument == 'rel':
                concept = block['argument'].split(' (')[0]

    elif block['operation'] == 'query':
        operation = 'query'
        argument = block['argument']   # need cleaning
        concept = ''

    elif block['operation'].startswith('verify'):
        operation = 'verify'
        if len(block['operation'].split())>1:
            argument = block['operation'].split()[1]    # argument can include rel
        else:
            argument = ''
        concept = block['argument']
        if argument == 'rel':
            concept = concept.split(' (')[0]

    elif block['operation'] == 'exist':
        operation = 'exist'
        argument = ''
        concept = ''

    elif block['operation'].startswith('same'):
        if len(block['dependencies']) == 1:
            operation = 'same_in1'
        elif len(block['dependencies']) == 2:
            operation = 'same_in2'
        else:
            raise NotImplementedError()
        if len(block['operation'].split())>1:
            argument = block['operation'].split()[1]
            # assert(block['argument']=='')
        else:
            argument = block['argument']
        concept = ''

    elif block['operation'].startswith('different'):
        if len(block['dependencies']) == 1:
            operation = 'different_in1'
            argument = block['argument']
            concept = ''
        elif len(block['dependencies']) == 2:
            operation = 'different_in2'
        else:
            raise NotImplementedError()
        if len(block['operation'].split())>1:
            argument = block['operation'].split()[1]
            # assert(block['argument']=='')
        else:
            argument = block['argument']
        concept = ''

    elif block['operation'] == 'common':    # answer is attribute
        operation = 'common'
        argument = ''
        concept = ''

    elif block['operation'] in ('and', 'or'):
        operation = block['operation']
        argument = ''
        concept = ''

    else:
        raise ValueError('Unsupported operation {}'.format(block['operation']))

    inputs = block['dependencies']
    return operation, argument, concept, inputs


def gqa_to_nsclseq(gqa_program):
    nscl_program = list()
    input_map = list()

    for block_id, block in enumerate(gqa_program):
        op, arg, concept, inputs = get_gqa_block_op(block)
        current = None
        # new_inputs = [input_map[i] for i in inputs]
        new_inputs = [input_map[i] if i<len(input_map) else input_map[-1] for i in inputs ] # handle predicted errors
        if op == 'select':
            current = dict(op='select', attr='', concept=concept, inputs=['_'])
        elif op == 'filter':
            current = dict(op='filter', attr=arg, concept=concept, inputs=new_inputs)
        elif op in ('relate_s', 'relate_o'):
            if concept != '_':
                current = dict(op='select', attr='', concept=concept, inputs=['_'])
                nscl_program.append(current)
                inp1 = len(nscl_program) - 1
            else:
                inp1 = '_'
            current = dict(op=op, rel=arg, inputs=[inp1, *new_inputs])
        elif op == 'relate_ae':
            # assert(concept != '_')
            current = dict(op='select', attr='', concept=concept, inputs=['_'])
            nscl_program.append(current)
            current = dict(op=op, attr=arg, inputs=[len(nscl_program) - 1, *new_inputs])
        elif op == 'choose_in1':
            if arg == 'rel':
                concept, rel_concept, so_ = concept.split(',')
                current = dict(op='select', attr='', concept=concept, inputs=['_'])
                nscl_program.append(current)
                if so_ == 's':
                    current = dict(op='query_rel_s', choices=rel_concept.split('|'), inputs=[len(nscl_program) - 1, *new_inputs])
                elif so_ == 'o':
                    current = dict(op='query_rel_o', choices=rel_concept.split('|'), inputs=[len(nscl_program) - 1, *new_inputs])
                else:
                    raise NotImplementedError()
            else:
                current = dict(op='query', attr=arg, choices=concept.split('|'), inputs=new_inputs)
        elif op == 'choose_in2':
            # TODO: may have problem here
            current = dict(op='choose', attr='', concept=arg, inputs=new_inputs, choices=[nscl_program[a]['concept'] for a in new_inputs])
        elif op == 'query':
            current = dict(op='query', attr=arg, choices=[], inputs=new_inputs)
        elif op == 'verify':
            if arg == 'rel':
                concept, rel_concept, so_ = concept.split(',')
                current = dict(op='select', attr='', concept=concept, inputs=['_'])
                nscl_program.append(current)
                if so_ == 's':
                    current = dict(op='verify_rel_s', rel=rel_concept, inputs=[len(nscl_program) - 1, *new_inputs])
                elif so_ == 'o':
                    current = dict(op='verify_rel_o', rel=rel_concept, inputs=[len(nscl_program) - 1, *new_inputs])
                else:
                    raise NotImplementedError()
            else:
                current = dict(op='verify', attr=arg, concept=concept, inputs=new_inputs)
        elif op == 'exist':
            current = dict(op='exist', inputs=new_inputs)
        elif op == 'same_in1':
            current = dict(op='same', attr=arg, inputs=new_inputs)
        elif op == 'same_in2':
            current = dict(op='query_ae', attr=arg, inputs=new_inputs)
        elif op == 'different_in1':
            current1 = dict(op='same', attr=arg, inputs=new_inputs)
            nscl_program.append(current1)
            current = dict(op='negate', inputs=[len(nscl_program) - 1])
        elif op == 'different_in2':
            current1 = dict(op='query_ae', attr=arg, inputs=new_inputs)
            nscl_program.append(current1)
            current = dict(op='negate', inputs=[len(nscl_program) - 1])
        elif op == 'common':
            current = dict(op='common', inputs=new_inputs)
        elif op in ('and'):
            current = dict(op='intersect', inputs=new_inputs)
        elif op in ('or'):
            current = dict(op='union', inputs=new_inputs)
        else:
            raise NotImplementedError()

        nscl_program.append(current)
        input_map.append(len(nscl_program) - 1)
    
    for current in nscl_program:
        # TODO: hacky to deal with special attrs here
        if 'attr' in current:
            attr_ = current['attr']
            if attr_ == 'type' or attr_ == 'sportActivity':
                current['attr'] = 'sport'
            elif attr_ == '' or attr_ == 'None':
                current['attr'] = 'void'
            elif attr_ == 'face expression':
                current['attr'] = 'face'
            elif attr_ in ['15', '18', '31', '16', '24', '25', '27']:
                current['attr'] = 'void'

    return nscl_program


choose_map = {
    'younger': ['young', 'age'], 
    'older': ['old', 'age'], 
    'healthier': ['healthy', ''], 
    'less_healthy': ['not(healthy)', ''], 
    'taller': ['tall', 'height'], 
    'shorter': ['short', 'length'], 
    'smaller': ['small', 'size'], 
    'larger': ['large', 'size'], 
    'lower': ['low', ''],
    'bigger': ['big', 'size']
}

def post_process(current):
    if current['op'] == 'choose':
        current['attr'] = choose_map[current['concept']][1]
        current['concept'] = choose_map[current['concept']][0]
    if 'attr' in current:
        attr_ = current['attr']
        if attr_ == 'type' or attr_ == 'sportActivity':
            current['attr'] = 'sport'
        elif attr_ == '' or attr_ == 'None':
            current['attr'] = 'void'
        elif attr_ == 'face expression':
            current['attr'] = 'face'
        elif attr_ in ['15', '18', '31', '16', '24', '25', '27']:
            current['attr'] = 'void'
    return current
