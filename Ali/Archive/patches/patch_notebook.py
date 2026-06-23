import json

with open('Ali/sentiment_pipeline_old.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Fix FutureWarning
        new_source = []
        for line in cell['source']:
            if 'mi["paid"]     = mi["is_paid_partnership"].fillna(False)' in line:
                line = line.replace('fillna(False)', 'fillna(False).infer_objects(copy=False)')
            new_source.append(line)
        cell['source'] = new_source
        
        # Add safety checks for 'sent' in all visualization cells
        source_str = "".join(cell['source'])
        if ('=== Polarity over time' in source_str or 
            '=== Cross-validation' in source_str or 
            '=== Emotion validity' in source_str or 
            '=== Room-level' in source_str):
            
            if 'globals()' not in source_str:
                new_source = []
                for line in cell['source']:
                    if line.startswith('if "timestamp" not in sent.columns'):
                        new_source.append('if "sent" not in globals():\n')
                        new_source.append('    print("Please run the \'Visualization setup\' cell first to define \'sent\'.")\n')
                        new_source.append('elif "timestamp" not in sent.columns or sent["timestamp"].notna().sum() == 0:\n')
                    elif line.startswith('order = ["negative", "neutral", "positive"]'):
                        new_source.append('if "sent" not in globals():\n')
                        new_source.append('    print("Please run the \'Visualization setup\' cell first to define \'sent\'.")\n')
                        new_source.append('else:\n')
                        new_source.append('    ' + line)
                    elif line.startswith('def _s2c(x):') or line.startswith('num_cat  =') or line.startswith('agree    =') or line.startswith('cat_prop =') or line.startswith('num_prop =') or line.startswith('fig, ax =') or line.startswith('data =') or line.startswith('bp =') or line.startswith('for ') or line.startswith('ax[') or line.startswith('x =') or line.startswith('ct =') or line.startswith('fig.') or line.startswith('print('):
                        if '=== Cross-validation' in source_str:
                            new_source.append('    ' + line)
                        else:
                            new_source.append(line)
                    elif line.startswith('em = sent.groupby') and '=== Emotion validity' in source_str:
                        new_source.insert(-2, 'if "sent" not in globals():\n')
                        new_source.insert(-1, '    print("Please run the \'Visualization setup\' cell first to define \'sent\'.")\n')
                        new_source.insert(0, 'else:\n')
                        new_source.append('    ' + line)
                    elif '=== Emotion validity' in source_str and (line.startswith('ecols =') or line.startswith('ax[') or line.startswith('featcols =') or line.startswith('if featcols:') or line.startswith('    cor =') or line.startswith('           .corr()') or line.startswith('    ax[') or line.startswith('else:') or line.startswith('    ax[') or line.startswith('torder =') or line.startswith('data =') or line.startswith('bp =') or line.startswith('for p, t') or line.startswith('    p.set_facecolor') or line.startswith('fig.')):
                        new_source.append('    ' + line)
                    else:
                        new_source.append(line)
                # For Polarity over time we specifically modified it earlier
                if '=== Polarity over time' in source_str:
                    pass # Handled by the first `if`
                
        # To avoid overly complex AST modification for python inside JSON via naive string matching,
        # A much simpler and robust approach is just to check at the top of the cell:
        # We will reset and do a simpler approach.
        pass

# Simpler approach:
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Fix FutureWarning
        new_source = []
        for line in cell['source']:
            if 'mi["paid"]     = mi["is_paid_partnership"].fillna(False)' in line:
                line = line.replace('fillna(False)', 'fillna(False).infer_objects(copy=False)')
            new_source.append(line)
        cell['source'] = new_source
        
        # Clear output if it contains an error or warning
        if 'outputs' in cell:
            new_outputs = []
            for out in cell['outputs']:
                if out.get('output_type') == 'error':
                    continue
                if out.get('name') == 'stderr' and 'FutureWarning' in ''.join(out.get('text', [])):
                    continue
                new_outputs.append(out)
            cell['outputs'] = new_outputs

        source_str = "".join(cell['source'])
        
        if '=== Polarity over time + sentiment composition over time ===' in source_str:
            new_source = []
            for line in cell['source']:
                if line.startswith('if "timestamp" not in sent.columns'):
                    new_source.append('if "sent" not in globals():\n')
                    new_source.append('    print("Please run the \'Visualization setup\' cell first to define \'sent\'.")\n')
                    new_source.append('elif "timestamp" not in sent.columns or sent["timestamp"].notna().sum() == 0:\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('Ali/sentiment_pipeline_old.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
