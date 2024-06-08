function table_to_csv(source) {
    const columns = Object.keys(source.data);
    const mol_html = columns.indexOf("mol_html");
    const removed = columns.splice(mol_html, 1);
    const nrows = source.get_length();
    const lines = [columns.join(',')];
    const idx = source.selected.indices;

    if (idx.length ==0) {
        for (let i = 0; i < nrows; i++) {
            let row = [];
            for (let j = 0; j < columns.length; j++) {
                const column = columns[j]
                if (column != "mol_html") {
                    row.push(source.data[column][i].toString())
                    }
                }
                lines.push(row.join(','))
            }
            return lines.join('\n').concat('\n')
        } 
    else {
        for (let i = 0; i < idx.length; i++) {
            let row = [];
            for (let j = 0; j < columns.length; j++) {
                const column = columns[j]
                if (column != "mol_html") {
                    row.push(source.data[column][idx[i]].toString())
                    }
                }
                lines.push(row.join(','))
            }
            return lines.join('\n').concat('\n')
        }
}   
    
console.log(source.data)
const filename = 'data_result.csv'
const filetext = table_to_csv(source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}