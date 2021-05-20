tmux list-sessions | awk 'BEGIN{FS=":"}{print $1}' | grep worker | xargs -n 1 tmux kill-session -t
