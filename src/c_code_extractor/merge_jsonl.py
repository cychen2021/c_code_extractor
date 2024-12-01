import os
import click

@click.command()
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), help='Path to the output file')
@click.argument('input_pattern', type=str)
@click.option('range_', '--range', '-r', type=str, help='Range of the numbers')
def main(output, input_pattern, range_):
    start_s, end_s = range_.split('..')
    result_lines = []
    for i in range(int(start_s), int(end_s) + 1):
        with open(input_pattern.replace('%r', str(i))) as f:
            content = f.read()
        lines = content.splitlines()
        result_lines.extend(lines)
    with open(output, 'w') as f:
        f.write('\n'.join(result_lines))

if __name__ == '__main__':
    main()
