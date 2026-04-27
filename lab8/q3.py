def ordinal_encode(column):
    unique_values = []

    for value in column:
        if value not in unique_values:
            unique_values.append(value)

    mapping = {value: index for index, value in enumerate(unique_values)}

    encoded = [mapping[value] for value in column]

    return encoded, mapping


def one_hot_encode(column):
    unique_values = []

    for value in column:
        if value not in unique_values:
            unique_values.append(value)

    encoded = []

    for value in column:
        row = []
        for category in unique_values:
            if value == category:
                row.append(1)
            else:
                row.append(0)
        encoded.append(row)

    return encoded, unique_values