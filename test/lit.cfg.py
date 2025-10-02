import os
import lit.formats

config.name = "simt-opt"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

config.substitutions.append(('%simt-opt', config.simt_opt_executable))
config.substitutions.append(('%mlir-file-check', config.mlir_file_check))
