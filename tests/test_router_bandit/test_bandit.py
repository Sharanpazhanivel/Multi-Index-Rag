"""Test bandit components."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.router.bandit import LinUCB, BanditRouter, get_action_retrievers
from src.schema import RouterDecision


def test_linucb_select_and_update() -> None:
    dim = 4
    num_actions = 3
    bandit = LinUCB(dim=dim, num_actions=num_actions, alpha=0.5)
    ctx = [0.1, 0.2, 0.3, 0.4]
    a = bandit.select(ctx)
    assert a in range(num_actions)
    bandit.update(ctx, a, reward=1.0)
    a2 = bandit.select(ctx)
    assert a2 in range(num_actions)


def test_bandit_router_linucb_route() -> None:
    # No embed_fn: context is zeros + prefs
    router = BanditRouter(strategy="linucb", num_actions=5, context_dim=386)
    d = router.route("test query")
    assert isinstance(d, RouterDecision)
    assert d.action_id in range(5)
    assert d.retriever_names == get_action_retrievers(d.action_id)


def test_bandit_router_epsilon_greedy_route() -> None:
    router = BanditRouter(strategy="epsilon_greedy", num_actions=5, epsilon=0.0)
    d = router.route("test query")
    assert isinstance(d, RouterDecision)
    assert d.action_id in range(5)
