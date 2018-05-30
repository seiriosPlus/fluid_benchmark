#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WARNING: _dist_transpile_if_necessary need to rewrite codes to accept cluster_2x4.sh
"""

    def _dist_transpile_if_necessary(self, optimize_ops, params_grads):
        self._transpile_nccl2_dist()
        if self.nccl_id_var != None:
            return

        if "PADDLE_TRAINING_ROLE" not in os.environ:
            return

        # the port of all pservers, needed by both trainer and pserver
        ip = os.getenv("PADDLE_IP")
        ports = os.getenv("PADDLE_PSERVER_PORTS")
        # comma separated ips of all pservers, needed by trainer and
        # pserver
        eplist = []
        for port in ports.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        # total number of workers/trainers in the job, needed by
        # trainer and pserver
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        # the unique trainer id, starting from 0, needed by trainer
        # only
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self.chief = self.trainer_id == 0
        # the role, should be either PSERVER or TRAINER
        training_role = os.getenv("PADDLE_TRAINING_ROLE")
        with self._prog_and_scope_guard():
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(
                trainer_id, pservers=pserver_endpoints, trainers=trainers)
            if training_role == "PSERVER":
                if self.checkpoint:
                    self.checkpoint.is_pserver = True
                # the IP of the local machine, needed by pserver only
                current_endpoint = ip+":"+os.getenv("CUR_PORT")

                self.train_program = t.get_pserver_program(current_endpoint)
                self.startup_program = t.get_startup_program(current_endpoint,
                                                             self.train_program)
            elif training_role == "TRAINER":
                self.train_program = t.get_trainer_program()
            else:
                raise ValueError(
                    'TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
                )