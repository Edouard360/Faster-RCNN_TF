from lxml import etree

def write_xml(symbols, bboxs, width, height, filename='figure1.xml'):
    root = etree.Element('annotation')

    etree.SubElement(root, "folder")
    etree.SubElement(root, "filename")
    etree.SubElement(root, "path")

    source = etree.SubElement(root, "source")
    database = etree.SubElement(source, "database")
    database.text = " "

    size = etree.SubElement(root, "size")
    w = etree.SubElement(size, "width")
    w.text = str(width)
    h = etree.SubElement(size, "height")
    h.text = str(height)
    depth = etree.SubElement(size, "depth")
    depth.text = str(1)

    segmented = etree.SubElement(root, "segmented")
    segmented.text = " "

    for s, bbox in zip(symbols, bboxs):
        object_balise = etree.SubElement(root, "object")
        name = etree.SubElement(object_balise, "name")
        s = s.split('\\')[-1].split('$')[0]
        name.text = s
        pose = etree.SubElement(object_balise, "pose")
        pose.text = "Unspecified"
        truncated = etree.SubElement(object_balise, "truncated")
        truncated.text = "0"
        difficult = etree.SubElement(object_balise, "difficult")
        difficult.text = "0"
        bndbox = etree.SubElement(object_balise, "bndbox")
        xmin = etree.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox[0]))
        ymin = etree.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox[1]))
        xmax = etree.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox[2]))
        ymax = etree.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox[3]))

    et = etree.ElementTree(root)
    et.write(filename, pretty_print=True)
    print("XML file written")
    return root