local sync = require('lib/sync')
local Project = require('lib/project')
local terminal = require('lib/terminal')

vim.api.nvim_create_autocmd('VimLeavePre', {
  callback = function()
    require('plenary.job'):new({
      command = 'killall',
      args = { 'lsyncd' },
      cwd = vim.loop.cwd()
    }):start()
  end
})

return Project.remote({
  name = 'inference-endpoints',
  sync_target = sync.cluster_target({
    clusters = Project.presets.nvidia_clusters,
    project_subdir = 'mlperf/endpoints.git/',
    use_worktrees = true,
    excludes = {
      '.venv*', 'venv*', '*.pyc', '__pycache__', '.ruff*', 'htmlcov', '.*_cache', 'outputs', 'logs', '.mypy_cache'
    },
  }),
  commands = {
  },
})
