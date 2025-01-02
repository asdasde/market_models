import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
from tabulate import tabulate
from io import StringIO

scraper = cloudscraper.create_scraper()
dfs = []
for i in range(192, 233):

    url = f'https://postal-codes.cybo.com/poland/?p={i}'

    source_html = BeautifulSoup(
            scraper.get(url).text,
            "lxml",
        ).select("table.paleblue")[-1]

    columns = [
        "postal_code", "city", "voivodeship", "postal_code_population", "postal_code_area",
    ]
    html_table = StringIO(str(source_html).replace("%", ""))
    tables = [table.iloc[:,:5] for table in pd.read_html(html_table)]
    df = pd.concat(tables)
    print(df)
    df.columns = columns
    df.drop(df.tail(1).index, inplace=True)
    dfs.append(df)
    res = pd.concat(dfs)
    print(i)
    res.to_csv("poland_postal_codes_1.csv", index=False)