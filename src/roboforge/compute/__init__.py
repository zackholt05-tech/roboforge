"""Remote compute (Modal): plans, budget gate, cost ledger, retrieval and profiles (Session 5).

* :mod:`~roboforge.compute.pricing`: posted list prices and cost arithmetic (estimates).
* :mod:`~roboforge.compute.plan`: launch plans, the launch rules and the ledger events.
* :mod:`~roboforge.compute.approval`: owner approvals read from the experiment ledger.
* :mod:`~roboforge.compute.cost_log`: the append-only cost/time ledger.
* :mod:`~roboforge.compute.remote`: job bodies shared by Modal and its local simulation.
* :mod:`~roboforge.compute.retrieve`: hash-validated installation of a downloaded run.
* :mod:`~roboforge.compute.profile`: training-throughput profile per family and device.
* :mod:`~roboforge.compute.estimate`: job and main-suite cost/time projections.

No module here imports the Modal SDK, which stays outside the project venv (D-045).
``infra/modal_app.py`` is the only Modal entrypoint.
"""
