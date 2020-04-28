import pandas as pd
import pathlib as pt
import click

@click.command()
@click.argument('path', type=click.STRING)
@click.option('--s', default='latex_copyable')
def main(path, s):
    csv_path = pt.Path(path)
    save_path = csv_path.parent/f"{s}.csv"
    assert csv_path.exists(), "the provided csv path does not exist"
    assert "csv" in csv_path.suffix, "file is not a csv"
    df = pd.read_csv(csv_path)
    output_string = ''

    per_item = False

    for key in df:
        data = df[key]
        if "Unnamed" not in key:
            output_string += key
            for item in data:
                if '(' in item:
                    # convert the data to the correct format
                    item = item.strip('(')
                    item = item.strip(')')
                    item = item.split(',')
                    item = [float(i) for i in item]
                    output_string += ', '
                    output_string += '%0.3f (%0.3f)'%(item[0], item[1])
                else:
                    item = float(item)
                    output_string += ', '
                    output_string += '%0.3f'%item
        else:
            nfolds = len(data) - 1
            output_string += ' '
            for fold_id in range(nfolds):
                output_string += ', fold %d'%(fold_id+1)
            output_string += ', average'

        output_string += '\n'

    with open(save_path, 'w') as f:
        f.write(output_string)

if __name__ == "__main__":
    main()