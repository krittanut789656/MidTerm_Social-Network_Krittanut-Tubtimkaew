"""
run_pipeline.py -- Full pipeline entry point.

Usage:
    python run_pipeline.py           # run all phases (2 + 3 + 4-sim + 6)
    python run_pipeline.py --phase 2 # data download, cleaning, features
    python run_pipeline.py --phase 3 # edge construction
    python run_pipeline.py --phase 4 # GDS simulation (networkx, no Neo4j needed)
    python run_pipeline.py --phase 6 # report generation
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def phase2():
    log.info("\n[PHASE 2 - Step 1/3] Downloading raw data...")
    from src.data_download import run_download
    run_download()

    log.info("\n[PHASE 2 - Step 2/3] Cleaning and aligning to weekly frequency...")
    from src.data_cleaning import run_cleaning
    run_cleaning()

    log.info("\n[PHASE 2 - Step 3/3] Transforming to returns and building features...")
    from src.feature_engineering import run_feature_engineering
    final = run_feature_engineering()
    log.info("Phase 2 complete: %d rows x %d cols", final.shape[0], final.shape[1])
    return final


def phase3():
    log.info("\n[PHASE 3 - Step 1/2] Regime detection...")
    from src.regime_detection import run_regime_detection
    run_regime_detection()

    log.info("\n[PHASE 3 - Step 2/2] Graph edge construction...")
    from src.graph_edges import run_graph_edges
    run_graph_edges()
    log.info("Phase 3 complete.")


def phase4():
    log.info("\n[PHASE 4] GDS simulation (networkx -- no Neo4j required)...")
    from src.gds_simulation import run_gds_simulation
    run_gds_simulation()
    log.info("Phase 4 complete.")


def phase6():
    log.info("\n[PHASE 6] Generating report draft...")
    from src.report_generator import run_report_generator
    md_path, docx_path = run_report_generator()
    log.info("Report saved: %s", md_path)
    log.info("Report saved: %s", docx_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase only (2, 3, 4, or 6). Default: run all.")
    args = parser.parse_args()

    log.info("=" * 62)
    log.info("  Thai Bank Graph Analysis -- Full Pipeline")
    log.info("=" * 62)

    if args.phase == 2:
        phase2()
    elif args.phase == 3:
        phase3()
    elif args.phase == 4:
        phase4()
    elif args.phase == 6:
        phase6()
    else:
        phase2()
        phase3()
        phase4()
        phase6()

    log.info("\n" + "=" * 62)
    log.info("  Pipeline complete.")
    log.info("  Next: streamlit run app.py  (all 9 pages ready)")
    log.info("  Optional (real Neo4j GDS):")
    log.info("    python src/neo4j_loader.py")
    log.info("    python src/gds_algorithms.py")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
