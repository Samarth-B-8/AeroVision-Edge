def parse_annotation_line(line):
    # Step 1: Remove whitespace/newline and split the line at commas
    values = line.strip().split(",")

    # Step 2: Convert each value from string to integer
    x = int(values[0])
    y = int(values[1])
    width = int(values[2])
    height = int(values[3])
    score = int(values[4])
    category = int(values[5])
    truncation = int(values[6])
    occlusion = int(values[7])

    # Step 3: Calculate the bottom-right coordinates
    x_max = x + width
    y_max = y + height

    # Step 4: Put everything into a dictionary
    annotation = {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "x_max": x_max,
        "y_max": y_max,
        "score": score,
        "category": category,
        "truncation": truncation,
        "occlusion": occlusion,
    }

    return annotation
