import re
from bs4 import BeautifulSoup


def build_citation_map(bbl_file):
    with open(bbl_file) as f:
        content = f.read()
    bib_items = re.findall(r"\\bibitem(?:\[\{(.*?)\}\])?\{(.*?)\}", content, flags=re.DOTALL)
    keys = [bib_items[i][1] for i in range(len(bib_items))]
    return {key: idx + 1 for idx, key in enumerate(keys)}


def fix_citations(html_file, bbl_file):
    """
    Fix citations in the HTML file by replacing missing citations with links to the bibliography.
    No replacement will be made if there is no 'ltx_missing_citation' class in the HTML file.

    Args:
        html_file (str): Path to the HTML file.
        bbl_file (str): Path to the BBL file.
    """

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    citation_map = build_citation_map(bbl_file)

    for span in soup.find_all("span", {"class": "ltx_missing_citation"}):
        key = span.text.strip()
        if key in citation_map:
            new_tag = soup.new_tag("a", href=f"#bib.{key}", **{"class": "ltx_ref"})
            new_tag.string = f"{citation_map[key]}"
            span.replace_with(new_tag)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print(f"✅ Fixed citations in {html_file}")


def main():
    bbl_file = "/home/ninziwei/lyj/nougat/__test_new/src/2303.00058/main.bbl"
    html_file = "/home/ninziwei/lyj/nougat/__test_new/html/2303.00058/2303.00058.html"
    citation_map = build_citation_map(bbl_file)
    fix_citations(html_file, citation_map)


if __name__ == "__main__":
    main()
