"""
Parse the FPStalker MySQL dump into a tidy CSV aligned to our feature names.

The dump is a phpMyAdmin export of one table (extensionDataScheme) with
multi-row INSERT statements. Large text fields (canvas/webgl/fonts) contain
commas, parentheses, and semicolons, so we parse with a small state machine that
tracks single-quote strings and backslash escapes, skipping quickly through long
quoted blobs.

We keep only the columns that align across datasets (we use the *Hashed* variants
of canvas/webgl/fonts), renamed to the Li & Cao feature names.

Output: data/raw/fpstalker/fingerprints.csv (tab-separated)
"""

import sys, os, csv, time

SQL = "data/raw/fpstalker/tableFingerprints.sql"
OUT = "data/raw/fpstalker/fingerprints.csv"

# FPStalker column -> our feature name (only columns we keep)
COLMAP = {
    "id": "browser_id",
    "creationDate": "creation_date",
    "userAgentHttp": "agent",
    "encodingHttp": "encoding",
    "languageHttp": "language",
    "platformJS": "fp2_platform",
    "cookiesJS": "cookie",
    "dntJS": "doNotTrack",
    "timezoneJS": "timezone",
    "resolutionJS": "resolution",
    "vendorWebGLJS": "fp2_webglvendoe",
    "rendererWebGLJS": "gpu",
    "pluginsJSHashed": "plugins",
    "canvasJSHashed": "canvastest",
    "webGLJsHashed": "fp2_webgl",
    "fontsFlashHashed": "jsFonts",
    "osDetailed": "os",
    "browserDetailed": "browser",
    "browserVersion": "browserversion",
}


def split_columns(col_str):
    """Parse the column list ``(a, b, ...)`` into a list of names."""
    return [c.strip().strip("`") for c in col_str.split(",")]


def parse_value_tuples(s):
    """Yield one list-of-fields per (...) tuple in a VALUES clause.

    A string-external semicolon ends the statement. Inside single-quoted
    strings we honor backslash escapes, so semicolons, commas, and parens
    inside blobs (e.g. the plugin list) are handled correctly.
    """
    i, n = 0, len(s)
    while i < n:
        # find next tuple opener; stop at a string-external ';'
        while i < n and s[i] not in "(;":
            i += 1
        if i >= n or s[i] == ";":
            return
        i += 1  # skip '('
        fields = []
        buf = []
        while i < n:
            c = s[i]
            if c == "'":
                buf.append(c)
                i += 1
                while i < n:
                    d = s[i]
                    if d == "\\":
                        buf.append(s[i:i+2]); i += 2; continue
                    buf.append(d); i += 1
                    if d == "'":
                        break
                continue
            if c == ",":
                fields.append("".join(buf)); buf = []; i += 1; continue
            if c == ")":
                fields.append("".join(buf)); i += 1
                yield fields
                break
            buf.append(c); i += 1


def clean(val):
    """Turn a raw SQL field token into a Python string value."""
    v = val.strip()
    if v == "NULL":
        return ""
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        v = v[1:-1]
        # unescape common MySQL escapes
        v = v.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
        v = v.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    return v


def main(test=False):
    t0 = time.time()
    keep_cols = list(COLMAP.keys())
    out_names = [COLMAP[c] for c in keep_cols]
    rows_written = 0

    with open(SQL, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"Read {len(text)/1e6:.0f} MB of SQL in {time.time()-t0:.1f}s")

    out_f = None if test else open(OUT, "w", newline="")
    writer = None if test else csv.writer(out_f, delimiter="\t")
    if writer:
        writer.writerow(out_names)

    # iterate over INSERT statements
    marker = "INSERT INTO `extensionDataScheme`"
    pos = 0
    n_insert = 0
    col_index = None
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        # column list between first '(' and ') VALUES'
        lp = text.find("(", start)
        vkey = text.find(") VALUES", lp)
        col_str = text[lp+1:vkey]
        cols = split_columns(col_str)
        if col_index is None:
            col_index = {name: idx for idx, name in enumerate(cols)}
            missing = [c for c in keep_cols if c not in col_index]
            if missing:
                print(f"warning: missing columns {missing}")
        # VALUES clause up to the next INSERT statement (or EOF); the parser
        # itself stops at the string-external ';' that ends this statement.
        vstart = vkey + len(") VALUES")
        nxt = text.find(marker, vstart)
        stmt_end = nxt if nxt != -1 else len(text)
        values_blob = text[vstart:stmt_end]
        pos = stmt_end
        n_insert += 1

        for fields in parse_value_tuples(values_blob):
            if len(fields) != len(cols):
                continue  # malformed, skip
            rec = [clean(fields[col_index[c]]) for c in keep_cols]
            if writer:
                writer.writerow(rec)
            rows_written += 1
            if test and rows_written <= 3:
                print("sample row:", dict(zip(out_names, rec))["agent"][:60],
                      "| canvas=", dict(zip(out_names, rec))["canvastest"][:12],
                      "| gpu=", dict(zip(out_names, rec))["gpu"][:25])
            if test and rows_written >= 5000:
                break
        if test and rows_written >= 5000:
            break

    if out_f:
        out_f.close()
    print(f"Parsed {n_insert} INSERT statements, {rows_written:,} fingerprint rows "
          f"in {time.time()-t0:.1f}s")
    if not test:
        print(f"output: {OUT}")


if __name__ == "__main__":
    main(test="--test" in sys.argv)
