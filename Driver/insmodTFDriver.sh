#!/bin/bash

base_dir=$(
  cd "$(dirname "$0")" || exit
  pwd
)

cd "$base_dir" || exit

operator=${1:-"insmod"}
ip=${2:-"none"}
driver_dir="${base_dir}/driver/tfacc2"
helper_name="tf_hugepage_register"

build_and_install_helper() {
  if ! (cd "${driver_dir}" && make "${helper_name}"); then
    echo "Failed to build ${helper_name}." >&2
    return 1
  fi
  if ! install -m 0755 "${driver_dir}/${helper_name}" "/usr/bin/${helper_name}"; then
    echo "Failed to install ${helper_name} to /usr/bin." >&2
    return 1
  fi
  echo "${helper_name} installed to /usr/bin/${helper_name}."
}

build_and_install_driver() {
  local build_output_dir="${driver_dir}/result"

  if [[ ${ip} == "none" ]]; then
    (cd "${driver_dir}" && ./build_driver.sh) || return 1
  else
    (cd "${driver_dir}" && ./build_driver.sh "${ip}") || return 1
    build_output_dir="${build_output_dir}/${ip}"
  fi

  local helper_path="${build_output_dir}/${helper_name}"
  if [[ ! -x ${helper_path} ]]; then
    echo "Failed to find built helper: ${helper_path}" >&2
    return 1
  fi

  if ! install -m 0755 "${helper_path}" "/usr/bin/${helper_name}"; then
    echo "Failed to install ${helper_name} to /usr/bin." >&2
    return 1
  fi

  echo "${helper_name} installed to /usr/bin/${helper_name}."
}

if [[ ${operator} == "insmod" ]]; then
  if lsmod | grep -q tfacc2; then
    echo "tfacc2 has already insmod."
    if ! build_and_install_helper; then
      exit 1
    fi
  else
    if ! modprobe tfacc2 >/dev/null 2>&1; then
      if ! build_and_install_driver; then
        echo "tfacc2 build or install failed." >&2
        exit 1
      fi

      if ! modprobe tfacc2; then
        echo "tfacc2 insmod failed." >&2
        exit 1
      fi
      echo "tfacc2 insmod success."
    else
      echo "tfacc2 insmod success."
      if ! build_and_install_helper; then
        exit 1
      fi
    fi
  fi
elif [[ ${operator} == "rm" ]]; then
  if ! lsmod | grep -q tfacc2; then
    echo "tfacc2 has not insmod."
    exit 0
  fi
  modprobe -r tfacc2
  echo 'tfacc2 remove success.'
elif [[ ${operator} == "delete" ]]; then
  find /lib/modules/ -name "tfacc2.ko" -type f -delete
  echo 'tfacc2 delete success.'
else
  echo 'Nothing to happen.'
fi
