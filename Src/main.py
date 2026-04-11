from Framework.ContextSystem.ExternalContextBuilder import BuildExternalContext
from Framework.RealtimeFlow import (
    BuildMarketData,
    BuildSystemContext,
    LoadThresholds,
    PrintFinalDecision,
    main,
)
from Framework.Utility.Utility import GetJSTNow, GetJSTNowStr, LoadJson


# --------------------------------------------------
# main.py:
#   どこで:
#     アプリケーションの最上位エントリポイント
#   何をやるか:
#     互換性のために必要な公開関数だけを再エクスポートし、
#     実際の realtime loop 本体は Framework 側へ委譲する
# --------------------------------------------------
if __name__ == "__main__":
    main()
