"""Wire the re-fetched IG media features into (a) an enriched, multimodal-scoped
ig_posts dataset and (b) the heterogeneous graph as new author nodes + edges.

Source: ``Output/ig_media_features.parquet`` (from ig_refetch.py), keyed by
SHORTCODE / web pk. The graph + ig_posts are keyed by the Graph-API ``media_id``
(a DIFFERENT id-space). Validated join chain:

    features.shortcode  ->  ig_posts.permalink  ->  ig_posts.media_id (Graph)
                        ->  graph node  ig_media_<media_id>

Critical decisions (see the diagnostics that motivated them):
  * ERROR rows (HTTP 400 — deleted/restricted) are DROPPED from the enriched
    posts dataset. Their pre-existing graph media nodes are left untouched.
  * Live private-API engagement (like_count etc.) DISAGREES with the Graph-API
    snapshot in ig_posts (only ~0.5% of likes match; drift over time). We do NOT
    overwrite the Graph values — live counts are added as ``*_live`` columns.
  * All 287 tagged + 11 coauthor user-pks are NEW to nodes_author (0 overlap) —
    appended as fresh ``ig_author_<pk>`` instagram nodes; usernames preserved in
    a side attribute file (the core nodes_author schema has no username column).

Edge directions (consistent with existing edges_* conventions — subject = src):
    edges_tagged      : media  -[tagged]->      author   (post tags an account)
    edges_coauthored  : author -[coauthored]->  media    (account co-produced it)

Outputs (originals backed up to *.bak before any in-place change):
    Output/ig_posts_multimodal_enriched.parquet
    Output/Prepared Comments/HeteroGraph/edges_tagged.parquet
    Output/Prepared Comments/HeteroGraph/edges_coauthored.parquet
    Output/Prepared Comments/HeteroGraph/nodes_author_ig_tags.parquet   (attrs)
    Output/Prepared Comments/HeteroGraph/nodes_author.parquet           (appended)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("wire_ig_features")

FEATURES = Path("Output/ig_media_features.parquet")
POSTS = Path("../Data/ig_posts_cleaned.parquet")
HG = Path("Output/Prepared Comments/HeteroGraph")
OUT_POSTS = Path("Output/ig_posts_multimodal_enriched.parquet")

PLATFORM = "instagram"
# features columns whose names collide with ig_posts → suffix to keep both.
RENAME = {
    "media_type": "media_type_api",
    "product_type": "product_type_api",
}
# The live private-API engagement snapshot is intentionally NOT carried into the
# enriched dataset: it is a later snapshot that disagrees with the Graph-API
# engagement already in ig_posts (use Graph reach/saved/views/total_interactions
# as the single consistent source). reshare_count/view_count were ~empty anyway.
LIVE_ENGAGEMENT = [
    "like_count", "comment_count", "play_count", "view_count",
    "reshare_count", "like_and_view_counts_disabled", "share_count_disabled",
]
DROP_FROM_FEATURES = ["ig_code", "fetch_status", "fetched_at", *LIVE_ENGAGEMENT]


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if p.exists() and not bak.exists():
        shutil.copy2(p, bak)
        LOGGER.info("backed up %s -> %s", p.name, bak.name)


def _shortcode(permalink: pd.Series) -> pd.Series:
    return permalink.str.extract(r"/(?:reel|p|tv)/([^/]+)/")[0]


def build_enriched_posts(feat_ok: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts["shortcode"] = _shortcode(posts["permalink"])
    feats = feat_ok.drop(columns=DROP_FROM_FEATURES).rename(columns=RENAME)
    # inner join: only multimodal posts that re-fetched OK survive (errors dropped)
    enr = posts.merge(feats, on="shortcode", how="inner", validate="one_to_one")
    LOGGER.info("enriched posts: %d rows (%d posts × OK fetch)", len(enr), len(feat_ok))
    return enr


def build_graph_objects(enr: pd.DataFrame):
    """Return (edges_tagged, edges_coauthored, new_author_nodes, author_attrs)."""
    enr = enr.copy()
    enr["media_node"] = "ig_media_" + enr["media_id"].astype(str)

    tagged_rows, coauth_rows, attrs = [], [], {}

    def _pairs(ids, names):
        ids = list(ids) if ids is not None else []
        names = list(names) if names is not None else []
        names = names + [None] * (len(ids) - len(names))
        return zip(ids, names)

    for _, r in enr.iterrows():
        mnode = r["media_node"]
        for pk, uname in _pairs(r["tagged_user_ids"], r["tagged_usernames"]):
            anode = f"ig_author_{pk}"
            tagged_rows.append({"src_media_id": mnode, "dst_author_id": anode,
                                "dst_username": uname, "platform": PLATFORM})
            attrs.setdefault(anode, {"username": uname, "roles": set()})["roles"].add("tagged")
        for pk, uname in _pairs(r["coauthor_ids"], r["coauthors"]):
            anode = f"ig_author_{pk}"
            coauth_rows.append({"src_author_id": anode, "dst_media_id": mnode,
                                "src_username": uname, "platform": PLATFORM})
            a = attrs.setdefault(anode, {"username": uname, "roles": set()})
            a["roles"].add("coauthor")
            if uname and not a["username"]:
                a["username"] = uname

    edges_tagged = pd.DataFrame(tagged_rows)
    edges_coauth = pd.DataFrame(coauth_rows)
    author_attrs = pd.DataFrame([
        {"author_id": k, "username": v["username"],
         "platform": PLATFORM, "roles": ",".join(sorted(v["roles"]))}
        for k, v in attrs.items()
    ])
    new_author_nodes = author_attrs[["author_id", "platform"]].copy()
    return edges_tagged, edges_coauth, new_author_nodes, author_attrs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    feat = pd.read_parquet(FEATURES)
    posts = pd.read_parquet(POSTS)
    feat_ok = feat[feat["fetch_status"] == "ok"].copy()
    n_err = len(feat) - len(feat_ok)
    LOGGER.info("features: %d (%d ok, %d errors dropped)", len(feat), len(feat_ok), n_err)

    # 1) Enriched posts -----------------------------------------------------
    enr = build_enriched_posts(feat_ok, posts)
    OUT_POSTS.parent.mkdir(parents=True, exist_ok=True)
    enr.to_parquet(OUT_POSTS, index=False)
    LOGGER.info("wrote %s (%d rows, %d cols)", OUT_POSTS, len(enr), enr.shape[1])

    # 2) Graph objects ------------------------------------------------------
    edges_tagged, edges_coauth, new_nodes, attrs = build_graph_objects(enr)

    na_path = HG / "nodes_author.parquet"
    nodes_author = pd.read_parquet(na_path)
    existing = set(nodes_author["author_id"].astype(str))
    truly_new = new_nodes[~new_nodes["author_id"].isin(existing)].drop_duplicates("author_id")

    _backup(na_path)
    nodes_author_updated = pd.concat([nodes_author, truly_new], ignore_index=True)
    nodes_author_updated.to_parquet(na_path, index=False)

    edges_tagged.to_parquet(HG / "edges_tagged.parquet", index=False)
    edges_coauth.to_parquet(HG / "edges_coauthored.parquet", index=False)
    attrs.to_parquet(HG / "nodes_author_ig_tags.parquet", index=False)

    # 3) Report -------------------------------------------------------------
    LOGGER.info("=== GRAPH WIRED ===")
    LOGGER.info("nodes_author: %d -> %d (+%d new tagged/coauthor accounts)",
                len(nodes_author), len(nodes_author_updated), len(truly_new))
    LOGGER.info("edges_tagged:     %d (media -> tagged account)", len(edges_tagged))
    LOGGER.info("edges_coauthored: %d (coauthor -> media)", len(edges_coauth))
    _report(enr)


def _report(enr: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("NEW INFORMATION RECOVERED  (enriched posts: %d)" % len(enr))
    print("=" * 64)
    ms = enr["music_source"].value_counts().to_dict()
    print("Music source:", ms)
    lic = enr[enr["music_source"] == "licensed"]
    print(f"Licensed songs: {len(lic)} | distinct tracks: {lic['audio_id'].nunique()} "
          f"| distinct artists: {lic['artist'].nunique()}")
    if len(lic):
        print("  Top artists:", lic["artist"].value_counts().head(5).to_dict())
    orig = enr[enr["music_source"] == "original"]
    print(f"Original audio: {len(orig)} | distinct audio_cluster_id: {orig['audio_cluster_id'].nunique()}")
    print(f"Paid partnerships: {int(enr['is_paid_partnership'].fillna(False).sum())}")
    print(f"Posts w/ product tags: {int((enr['n_product_tags'] > 0).sum())} "
          f"| featured products: {int((enr['n_featured_products'] > 0).sum())}")
    print(f"Posts w/ tagged accounts: {int((enr['n_tagged'] > 0).sum())} "
          f"| total tags: {int(enr['n_tagged'].sum())}")
    print(f"Posts that are collaborations: {int((enr['n_coauthors'] > 0).sum())} "
          f"| total collab links: {int(enr['n_coauthors'].sum())}")
    coll = [c for r in enr["coauthors"] if r is not None for c in r]
    if coll:
        print("  Collaborators:", dict(pd.Series(coll).value_counts()))
    tg = [t for r in enr["tagged_usernames"] if r is not None for t in r]
    if tg:
        print("  Top tagged accounts:", pd.Series(tg).value_counts().head(8).to_dict())
    print("\n[note] live private-API engagement snapshot dropped; Graph-API "
          "engagement in ig_posts (reach/saved/views/total_interactions) is canonical.")


if __name__ == "__main__":
    main()